#engine v8   // must sit in the first few lines: PixInsight only scans a
             // short prologue for it, and a header comment long enough to
             // push it past ~line 20 leaves the script in ES5, where let
             // and arrow functions are a load error that -r mode discards
             // silently with exit 0 and no log.
/*
 * corona-hdr.js — merge the exposure ladder into one linear corona image, then
 * flatten its radial falloff.
 *
 * The corona spans a brightness range no single exposure can hold: the inner
 * corona saturates long before the outer streamers rise out of the noise. This
 * takes the per-level colour images, puts them on a common intensity scale
 * measured from their own overlap, and blends them so each pixel comes from the
 * longest exposure that has not clipped there.
 *
 * The second half is what actually makes the corona visible. Its brightness falls
 * off close to exponentially with radius, so no global stretch can show inner and
 * outer detail at once. Dividing out the measured radial profile removes the
 * falloff, leaving the structure; a fraction of the profile is then re-applied so
 * the result still reads as a corona rather than a flat disc.
 *
 * HDRComposition is deliberately not used: the merge is a handful of arithmetic
 * decisions and doing them here keeps them inspectable and reproducible.
 *
 *   -r="...corona-hdr.js,<config.json>,<logPath>"
 *
 * Config: { "images": [ { "path":..., "level":n, "cx":..., "cy":..., "r":... } ... ],
 *           "out":..., "outFlat":... }
 */


// Saturation feather. A stack is an average, so a pixel saturated in every frame
// lands near 1.0 while one saturated in half the frames lands lower; the ramp
// retires a level gradually instead of at a cliff.
const SAT_LO = 0.70;
const SAT_HI = 0.94;

// Noise floor feather: a short exposure carries almost no signal in the faint
// outer corona, and including it there would only add noise.
const SIG_LO = 0.0008;
const SIG_HI = 0.0060;

/*
 * Overlap window used to match two exposure levels.
 *
 * Separate bounds for the two frames, and deliberately wide. Fitting a slope
 * alone tolerates a narrow window, but fitting slope AND intercept needs real
 * dynamic range or the line is unconstrained: the first attempt reused a single
 * [0.02, 0.75] window, which overlaps only in a thin annulus where the shorter
 * exposure spans a factor of two, and the regression came back with a negative
 * slope and fell through to identity.
 *
 * The shorter frame only has to be above its own noise and unsaturated; the
 * longer one has to be unsaturated and carrying real signal.
 */
const FIT_SHORT_LO = 0.0020;
const FIT_SHORT_HI = 0.90;
const FIT_LONG_LO = 0.0050;
const FIT_LONG_HI = 0.85;

// Minimum span of the shorter frame's values across the accepted samples, as a
// ratio. Below this the intercept is not meaningfully constrained.
const FIT_MIN_SPAN = 3.0;

const FIT_STRIDE = 37;   // coprime with the row length: samples the whole frame

// How much of the measured radial profile to put back after flattening. 0 is a
// completely flat field that looks synthetic; 1 is the original falloff.
const PROFILE_RESTORE = 0.45;

// Inside this radius (in Moon radii) the lunar limb dominates the picture, so a
// level whose Moon has moved more than LIMB_TOL_PX from the reference time is
// excluded there — otherwise its displaced dark limb prints as a crescent.
// Outside it the corona is what matters and every level contributes.
// Contamination from a limb displaced by e reaches to about R + e, so that is
// where a level starts being admitted, reaching full weight LIMB_FEATHER_PX
// further out. Feathering matters: a hard cut switches every excluded level on
// at the same radius and leaves a visible arc there.
const LIMB_FEATHER_PX = 90.0;
const LIMB_TOL_PX = 1.5;

// Pixels a radius needs before its median counts as a profile measurement.
const MIN_RADIAL_SAMPLES = 2000;

function main()
{
   let cfgPath = jsArguments[0];
   let logPath = jsArguments[1];

   let logFile = new File;
   logFile.createForWriting( logPath );
   function log( s )
   {
      logFile.outTextLn( String( s ) );
      logFile.flush();
      console.writeln( String( s ) );
      console.flush();
   }

   try
   {
      let t0 = Date.now();
      // File.readTextFile does not exist; a missing PJSR API kills the script
      // silently with exit 0. utf8ToString on a ByteArray is the working idiom.
      let cfg = JSON.parse( File.readFile( cfgPath ).utf8ToString().replace( /^\uFEFF/, "" ) );
      let specs = cfg.images.slice().sort( ( a, b ) => a.level - b.level );

      /*
       * Reference frame.
       *
       * The Moon and the corona move relative to each other, so no single rigid
       * registration holds both. The output is defined to be the Moon frame at
       * one instant — the shortest exposure's capture time, since that is the
       * level that carries the limb, the chromosphere and the prominences.
       *
       * Every other level is then placed so its CORONA lands where the corona was
       * at that instant: Moon registration first, then the measured differential
       * drift taken back out. At the reference time the two frames coincide, so
       * the result has a sharp limb AND a sharp corona; the levels furthest in
       * time contribute only where the limb is not the subject, which is what the
       * gate below enforces.
       */
      let ref = specs[0];
      let tRef = (ref.t !== undefined) ? ref.t : 0;
      let drift = cfg.drift || { x: 0, y: 0 };
      let hasDrift = (drift.x !== 0 || drift.y !== 0);
      log( hasDrift
           ? "  differential drift (" + drift.x.toFixed( 4 ) + ", "
             + drift.y.toFixed( 4 ) + ") px/s, reference = level "
             + ref.level + " at t=" + tRef.toFixed( 1 )
           : "  no drift supplied — rigid Moon registration only" );

      let W = 0, H = 0;
      let validX0 = 0, validX1 = 0, validY0 = 0, validY1 = 0;
      let loaded = [];
      for ( let k = 0; k < specs.length; ++k )
      {
         let sp = specs[k];
         let wins = ImageWindow.open( sp.path );
         let win = wins[0];
         for ( let i = 1; i < wins.length; ++i )
            wins[i].forceClose();
         let img = win.mainView.image;
         if ( k == 0 ) { W = img.width; H = img.height; }
         else if ( img.width != W || img.height != H )
            throw new Error( "geometry mismatch on " + sp.path );

         let ch = [];
         for ( let c = 0; c < 3; ++c )
         {
            let a = new Float32Array( W*H );
            img.getSamples( a, new Rect( 0, 0, W, H ), c );
            ch.push( a );
         }
         win.forceClose();

         let dt = ((sp.t !== undefined) ? sp.t : 0) - tRef;
         // Prefer the shift measured by correlating this level against the
         // reference. Independent per-level Moon fits disagree by hundreds of
         // pixels at the extremes of the exposure ladder; the difference of two
         // such fits is only used when no measured registration was supplied.
         let baseX = (sp.dx !== undefined) ? sp.dx : (ref.cx - sp.cx);
         let baseY = (sp.dy !== undefined) ? sp.dy : (ref.cy - sp.cy);
         let dx = baseX - drift.x*dt;
         let dy = baseY - drift.y*dt;
         if ( Math.abs( dx ) > 0.01 || Math.abs( dy ) > 0.01 )
            for ( let c = 0; c < 3; ++c )
               ch[c] = translateImage( ch[c], W, H, dx, dy );
         // Track where every level still has real data. A shifted level runs off
         // the sensor on one side, and the boundary where it stops contributing
         // shows up as a step in the merge - the pixels there are built from
         // fewer levels than the rest of the frame.
         if ( dx > validX0 ) validX0 = dx;
         if ( dx < validX1 ) validX1 = dx;
         if ( dy > validY0 ) validY0 = dy;
         if ( dy < validY1 ) validY1 = dy;
         // How far this level's Moon now sits from the reference Moon.
         let limbErr = Math.sqrt( drift.x*drift.x + drift.y*drift.y )*Math.abs( dt );
         log( "  L" + sp.level + " " + sp.path.substring( sp.path.lastIndexOf( "/" ) + 1 )
              + " dt=" + dt.toFixed( 1 ) + "s translateImage(" + dx.toFixed( 2 ) + ", "
              + dy.toFixed( 2 ) + ") limb offset " + limbErr.toFixed( 1 ) + " px" );
         loaded.push( { level: sp.level, ch: ch, limbErr: limbErr } );
      }

      // ---- average images that share an exposure level ----
      let levels = [];
      for ( let i = 0; i < loaded.length; )
      {
         let j = i;
         while ( j < loaded.length && loaded[j].level == loaded[i].level )
            ++j;
         if ( j - i == 1 )
            levels.push( loaded[i] );
         else
         {
            // Chunks of one level are close in time; the group inherits the worst
            // limb offset among them.
            let worst = 0;
            for ( let k = i; k < j; ++k )
               if ( loaded[k].limbErr > worst ) worst = loaded[k].limbErr;
            let acc = [ new Float32Array( W*H ), new Float32Array( W*H ), new Float32Array( W*H ) ];
            for ( let k = i; k < j; ++k )
               for ( let c = 0; c < 3; ++c )
               {
                  let s = loaded[k].ch[c];
                  for ( let p = 0; p < W*H; ++p )
                     acc[c][p] += s[p];
               }
            let inv = 1/(j - i);
            for ( let c = 0; c < 3; ++c )
               for ( let p = 0; p < W*H; ++p )
                  acc[c][p] *= inv;
            log( "  averaged " + (j - i) + " images at level " + loaded[i].level
                 + " (worst limb offset " + worst.toFixed( 1 ) + " px)" );
            levels.push( { level: loaded[i].level, ch: acc, limbErr: worst } );
         }
         i = j;
      }
      log( "  " + levels.length + " exposure levels" );

      /* ---- put every level on the shortest exposure's scale ----
       *
       * The relation between two exposures is affine, not a ratio. Every frame
       * carries an additive pedestal - sensor offset plus sky - so a level reads
       * pedestal + exposure x signal, and a pure multiplicative scale can only
       * match two levels near wherever it was fitted. Measured on this data, a
       * ratio fitted near the limb overpredicts the longer exposure by 315% out
       * at 2.5 Moon radii, and the sign change on the way printed a ring at
       * 1.47 R. An affine fit holds to about 10% over the same span.
       *
       * Fitted per channel: the CFA channels do not share a pedestal, so folding
       * them into one number would leave a radial colour cast behind.
       */
      let A = [], B = [];
      for ( let c = 0; c < 3; ++c ) { A.push( [ 1.0 ] ); B.push( [ 0.0 ] ); }
      for ( let k = 1; k < levels.length; ++k )
      {
         for ( let c = 0; c < 3; ++c )
         {
            let f = ecFitAffine( levels[k - 1].ch[c], levels[k].ch[c], W*H );
            if ( !(f.a > 0) )
            {
               log( "  WARNING: level " + levels[k - 1].level + " -> " + levels[k].level
                    + " ch" + c + ": unusable overlap (n=" + f.n
                    + ", span=" + (f.span || 0).toFixed( 2 ) + "x) - assuming equal" );
               f = { a: 1, b: 0 };
            }
            else if ( c == 1 )
               log( "  fit ch1 used " + f.n + " samples spanning "
                    + f.span.toFixed( 1 ) + "x" );
            // Compose level k -> level 0: v0 = (vk - B)/A.
            A[c].push( A[c][k - 1]*f.a );
            B[c].push( f.b + f.a*B[c][k - 1] );
            if ( c == 1 )
               log( "  level " + levels[k].level + " = " + f.a.toFixed( 3 ) + " x level "
                    + levels[k - 1].level + " + " + f.b.toExponential( 3 )
                    + "  (cumulative A=" + A[c][k].toFixed( 2 )
                    + " B=" + B[c][k].toExponential( 3 ) + ")" );
         }
      }

      // ---- weighted merge ----
      let out = [ new Float32Array( W*H ), new Float32Array( W*H ), new Float32Array( W*H ) ];
      let wsum = new Float32Array( W*H );
      let n = W*H;

      for ( let k = 0; k < levels.length; ++k )
      {
         let g = levels[k].ch[1];
         let e = levels[k].limbErr || 0;
         // Where this level starts and finishes being admitted, by its own offset.
         let gated = e > LIMB_TOL_PX;
         let gLo = ref.r + e, gHi = gLo + LIMB_FEATHER_PX;
         if ( gated )
            log( "  level " + levels[k].level + " feathered in over r="
                 + gLo.toFixed( 0 ) + ".." + gHi.toFixed( 0 )
                 + " px (limb offset " + e.toFixed( 1 ) + " px)" );
         for ( let y = 0; y < H; ++y )
         {
            let dy = y - ref.cy, row = y*W;
            for ( let x = 0; x < W; ++x )
            {
               let p = row + x;
               let gate = 1;
               if ( gated )
               {
                  let dx = x - ref.cx;
                  gate = ecRamp( Math.sqrt( dx*dx + dy*dy ), gLo, gHi );
                  if ( gate <= 0 )
                     continue;
               }
               let w = gate*ecRamp( g[p], SIG_LO, SIG_HI )
                       *(1 - ecRamp( g[p], SAT_LO, SAT_HI ));
               if ( w <= 0 )
                  continue;
               wsum[p] += w;
               for ( let c = 0; c < 3; ++c )
                  out[c][p] += w*(levels[k].ch[c][p] - B[c][k])/A[c][k];
            }
         }
      }
      // Pixels no level could represent — saturated even at the shortest exposure,
      // i.e. the chromosphere and any beads — fall back to the shortest exposure.
      let orphan = 0;
      for ( let p = 0; p < n; ++p )
         if ( wsum[p] > 0 )
         {
            let iw = 1/wsum[p];
            for ( let c = 0; c < 3; ++c )
               out[c][p] *= iw;
         }
         else
         {
            orphan++;
            for ( let c = 0; c < 3; ++c )
               out[c][p] = levels[0].ch[c][p];
         }
      log( "  merged; " + orphan + " px ("
           + (100*orphan/n).toFixed( 3 ) + "%) fell back to the shortest exposure" );

      // Trim to the region every level covers, so the output has no boundary
      // where the number of contributing levels changes.
      let cx0 = Math.ceil( Math.max( 0, validX0 ) );
      let cy0 = Math.ceil( Math.max( 0, validY0 ) );
      let cx1 = Math.floor( W + Math.min( 0, validX1 ) );
      let cy1 = Math.floor( H + Math.min( 0, validY1 ) );
      let cw = cx1 - cx0, chh = cy1 - cy0;
      log( "  common coverage " + cw + "x" + chh + " at (" + cx0 + ", " + cy0
           + "), trimmed " + (W - cw) + "x" + (H - chh) + " px" );

      ecSave( ecCrop( out, W, H, cx0, cy0, cw, chh ), cw, chh, cfg.out, "corona_hdr" );
      log( "  saved " + cfg.out );

      // ---- radial flatten ----
      // Flatten on the trimmed frame, with the Moon centre moved into its
      // coordinates, so the radial profile is not measured through the seam.
      let flat = ecFlatten( ecCrop( out, W, H, cx0, cy0, cw, chh ), cw, chh,
                            ref.cx - cx0, ref.cy - cy0, ref.r, log );
      ecSave( flat, cw, chh, cfg.outFlat, "corona_flat" );

      // Record the Moon in the TRIMMED frame's own coordinates. Later stages
      // measure annuli about it, and the sidecars written by corona-combine.js
      // refer to the untrimmed frame.
      let side = new File;
      side.createForWriting( cfg.outFlat.replace( /\.xisf$/i, "" ) + "_moon.json" );
      side.outTextLn( JSON.stringify( { image: cfg.outFlat, width: cw, height: chh,
                                        cx: ref.cx - cx0, cy: ref.cy - cy0,
                                        radius: ref.r } ) );
      side.close();
      log( "  saved " + cfg.outFlat );

      log( "  done in " + ((Date.now() - t0)/1000).toFixed( 0 ) + " s" );
      log( "=== HDR OK ===" );
   }
   catch ( e )
   {
      log( "*** HDR FAILED: " + e.toString() );
      if ( e.stack )
         log( String( e.stack ) );
   }
   logFile.close();
}

/* Smooth 0->1 ramp; used so levels enter and leave the blend gradually. */
function ecRamp( v, lo, hi )
{
   if ( v <= lo ) return 0;
   if ( v >= hi ) return 1;
   let t = (v - lo)/(hi - lo);
   return t*t*(3 - 2*t);
}

/* Bilinear whole-image translation. Registration between exposure levels is pure
 * translation: the mount tracked, and the frames are seconds apart. */
function translateImage( src, W, H, dx, dy )
{
   let dst = new Float32Array( W*H );
   for ( let y = 0; y < H; ++y )
   {
      let sy = y - dy;
      let y0 = Math.floor( sy ), fy = sy - y0, y1 = y0 + 1;
      // Outside the source, leave zero rather than clamping to the edge pixel.
      // Shifts here reach 227 px, and replicating the border smeared a bright
      // strip down two sides of the merge - and because the strip is bright it
      // passed the signal test and was blended in as if it were real data. Zero
      // fails that test, so those pixels fall back to the unshifted reference.
      if ( y0 < 0 || y1 >= H ) continue;
      let r0 = y0*W, r1 = y1*W, drow = y*W;
      for ( let x = 0; x < W; ++x )
      {
         let sx = x - dx;
         let x0 = Math.floor( sx ), fx = sx - x0, x1 = x0 + 1;
         if ( x0 < 0 || x1 >= W ) continue;
         let a = src[r0 + x0] + (src[r0 + x1] - src[r0 + x0])*fx;
         let b = src[r1 + x0] + (src[r1 + x1] - src[r1 + x0])*fx;
         dst[drow + x] = a + (b - a)*fy;
      }
   }
   return dst;
}

/*
 * Least-squares affine fit long = a*short + b over pixels well exposed in both.
 *
 * The intercept is the point: it absorbs the difference in pedestal between the
 * two exposures. Without it the fit is forced through the origin and can only be
 * right over a narrow band of brightness.
 *
 * Outliers are trimmed once against the first fit, so prominences and the
 * chromosphere - which sit at the very top of the fitting window and are not
 * part of the smooth corona relation - cannot tilt the line.
 */
function ecFitAffine( shortC, longC, n )
{
   let xs = [], ys = [];
   let xlo = Infinity, xhi = -Infinity;
   for ( let p = 0; p < n; p += FIT_STRIDE )
   {
      let a = shortC[p], b = longC[p];
      if ( a > FIT_SHORT_LO && a < FIT_SHORT_HI && b > FIT_LONG_LO && b < FIT_LONG_HI )
      {
         xs.push( a ); ys.push( b );
         if ( a < xlo ) xlo = a;
         if ( a > xhi ) xhi = a;
      }
   }
   if ( xs.length < 100 || !(xhi > xlo*FIT_MIN_SPAN) )
      return { a: 0, b: 0, n: xs.length, span: (xlo > 0 ? xhi/xlo : 0) };

   function solve( idx )
   {
      let m = idx.length, sx = 0, sy = 0, sxx = 0, sxy = 0;
      for ( let i = 0; i < m; ++i )
      {
         let x = xs[idx[i]], y = ys[idx[i]];
         sx += x; sy += y; sxx += x*x; sxy += x*y;
      }
      let d = m*sxx - sx*sx;
      if ( !(Math.abs( d ) > 0) )
         return { a: 0, b: 0 };
      let a = (m*sxy - sx*sy)/d;
      return { a: a, b: (sy - a*sx)/m };
   }

   let all = [];
   for ( let i = 0; i < xs.length; ++i )
      all.push( i );
   let f = solve( all );
   if ( !(f.a > 0) )
      return f;

   let res = [];
   for ( let i = 0; i < xs.length; ++i )
      res.push( Math.abs( ys[i] - (f.a*xs[i] + f.b) ) );
   let s = res.slice().sort( ( u, v ) => u - v );
   let cut = 3*s[s.length >> 1] + 1e-9;
   let keep = [];
   for ( let i = 0; i < xs.length; ++i )
      if ( res[i] <= cut )
         keep.push( i );
   let r = (keep.length >= 100) ? solve( keep ) : f;
   r.n = xs.length; r.span = xhi/xlo;
   return r;
}

/*
 * Divide out the corona's radial brightness profile.
 *
 * The profile is a per-radius median of luminance, so streamers (which are local
 * excursions) do not pull it up. Inside the limb there is no corona to flatten,
 * so the profile is held at its limb value and the disc keeps its own darkness.
 */
function ecFlatten( rgb, W, H, cx, cy, moonR, log )
{
   let n = W*H;
   let rmax = Math.ceil( Math.sqrt( Math.max( cx, W - cx )*Math.max( cx, W - cx )
                                  + Math.max( cy, H - cy )*Math.max( cy, H - cy ) ) );

   // Per-radius median via a coarse histogram of log luminance.
   const BINS = 2048;
   let LMIN = -20, LMAX = 2;                     // log2 luminance range
   let hist = new Int32Array( (rmax + 1)*BINS ); // radius-major
   let cnt = new Int32Array( rmax + 1 );
   let lum = new Float32Array( n );
   for ( let y = 0; y < H; ++y )
   {
      let dy = y - cy, row = y*W;
      for ( let x = 0; x < W; ++x )
      {
         let p = row + x;
         let L = 0.25*rgb[0][p] + 0.60*rgb[1][p] + 0.15*rgb[2][p];
         lum[p] = L;
         let dx = x - cx;
         let r = Math.round( Math.sqrt( dx*dx + dy*dy ) );
         if ( r > rmax ) continue;
         let l2 = (L > 0) ? Math.log2( L ) : LMIN;
         let b = Math.round( (l2 - LMIN)/(LMAX - LMIN)*(BINS - 1) );
         if ( b < 0 ) b = 0; else if ( b >= BINS ) b = BINS - 1;
         hist[r*BINS + b]++;
         cnt[r]++;
      }
   }

   /*
    * Beyond the largest circle that fits in the frame, a radius is sampled only
    * where it clips a corner, so its median is drawn from a handful of pixels and
    * is not a profile at all. Dividing by it multiplied the corners up into a
    * bright band around two edges of the flattened image. Past the last radius
    * with enough support, the profile is frozen at its last trustworthy value -
    * the corners then get a constant gain, which is honest: there is no
    * measurement out there to correct with.
    */
   // Scan outward from the limb: the innermost radii legitimately have almost no
   // pixels (r = 0 is one pixel) and starting there would break out immediately.
   let rValid = rmax;
   for ( let r = Math.max( 1, Math.ceil( moonR ) ); r <= rmax; ++r )
      if ( cnt[r] < MIN_RADIAL_SAMPLES ) { rValid = r - 1; break; }

   let prof = new Float64Array( rmax + 1 );
   for ( let r = 0; r <= rValid; ++r )
   {
      if ( cnt[r] == 0 ) { prof[r] = (r > 0) ? prof[r - 1] : 1e-6; continue; }
      let half = cnt[r] >> 1, acc = 0, b = 0;
      for ( ; b < BINS - 1 && acc < half; ++b )
         acc += hist[r*BINS + b];
      prof[r] = Math.pow( 2, LMIN + b/(BINS - 1)*(LMAX - LMIN) );
   }
   for ( let r = rValid + 1; r <= rmax; ++r )
      prof[r] = prof[rValid];
   log( "  radial profile valid to r=" + rValid + " of " + rmax
        + " px; held constant beyond" );

   // Hold the profile flat inside the limb, and smooth it so pixel-scale noise in
   // the profile does not print as rings.
   let rIn = Math.max( 1, Math.round( moonR*0.99 ) );
   for ( let r = 0; r < rIn && r <= rmax; ++r )
      prof[r] = prof[rIn];
   let sm = new Float64Array( rmax + 1 );
   const K = 9;
   for ( let r = 0; r <= rmax; ++r )
   {
      let s = 0, c = 0;
      for ( let d = -K; d <= K; ++d )
      {
         let q = r + d;
         if ( q >= 0 && q <= rmax ) { s += prof[q]; c++; }
      }
      sm[r] = s/c;
   }

   let ref = sm[rIn] > 0 ? sm[rIn] : 1e-6;
   let out = [ new Float32Array( n ), new Float32Array( n ), new Float32Array( n ) ];
   for ( let y = 0; y < H; ++y )
   {
      let dy = y - cy, row = y*W;
      for ( let x = 0; x < W; ++x )
      {
         let p = row + x, dx = x - cx;
         let r = Math.round( Math.sqrt( dx*dx + dy*dy ) );
         if ( r > rmax ) r = rmax;
         let pr = sm[r] > 0 ? sm[r] : 1e-6;
         // Full division would flatten the corona into a uniform sheet; putting a
         // fraction of the profile back keeps the natural inner-to-outer gradient.
         let gain = ref/pr*Math.pow( pr/ref, PROFILE_RESTORE );
         for ( let c = 0; c < 3; ++c )
            out[c][p] = rgb[c][p]*gain;
      }
   }
   log( "  flattened about (" + cx.toFixed( 1 ) + ", " + cy.toFixed( 1 )
        + "), limb r=" + moonR + ", profile restored " + PROFILE_RESTORE );
   return out;
}

/* Extract a sub-rectangle from all three planes. */
function ecCrop( rgb, W, H, x0, y0, w, h )
{
   let out = [ new Float32Array( w*h ), new Float32Array( w*h ), new Float32Array( w*h ) ];
   for ( let c = 0; c < 3; ++c )
   {
      let src = rgb[c], dst = out[c];
      for ( let y = 0; y < h; ++y )
      {
         let s = (y0 + y)*W + x0, d = y*w;
         for ( let x = 0; x < w; ++x )
            dst[d + x] = src[s + x];
      }
   }
   return out;
}

function ecSave( rgb, W, H, path, id )
{
   let win = new ImageWindow( W, H, 3, 32, true, true, id );
   let v = win.mainView;
   v.beginProcess( UndoFlag.NoSwapFile );
   for ( let c = 0; c < 3; ++c )
   {
      v.image.selectedChannel = c;
      v.image.setSamples( rgb[c] );
   }
   v.image.resetSelections();
   v.endProcess();
   if ( !win.saveAs( path, false, false, false, false ) )
      throw new Error( "save failed: " + path );
   win.forceClose();
}

main();
