#engine v8   // keep within the first few lines: PixInsight only scans a short
             // prologue for this, and a longer header comment pushes it out of
             // range, leaving the script in ES5 where let and arrow functions
             // are a load error that -r mode discards silently with exit 0.

/*
 * tl-centres.js - pass 1 of the timelapse: find the Sun in every frame.
 *
 * Framing on the centroid of the lit region does not hold the Sun still. The
 * centroid of a crescent is not the centre of the disc it was cut from, and it
 * slides further off as the Moon covers more of the Sun, then jumps when the
 * filter comes off and the bright region becomes a corona. That is the bouncing.
 * What is actually stationary is the Sun's limb: a circle of known radius.
 *
 * FILTERED FRAMES - Kasa circle fit with a radius prior, following
 * analyzeDisk() in pix-planetary's gif-frames.js, which solves the same problem
 * for lunar crescents. Limb points come from a row scan taking the first and
 * last bright pixel per row; on a partial eclipse that yields the Sun's limb on
 * one side and the Moon's on the other, and the radius prior plus a shrinking
 * inlier tolerance rejects the wrong one. Their note applies directly here:
 * thin crescents need the prior, because terminator points pollute the fit.
 * Points are taken at a LOW threshold for the same reason they found - a soft
 * limb ramps over several pixels and a high crossing sits inside the true limb,
 * biasing the radius small.
 *
 * UNFILTERED FRAMES - no photosphere to fit, so the Moon's limb is found by
 * scoring candidate centres on the gradient of log brightness around a circle of
 * the Moon's radius. The two discs are nearly concentric during totality, so the
 * substitution costs at most (Rmoon - Rsun), about 13 px here, and what is left
 * is the real offset between the bodies - which is the thing worth watching.
 *
 * Quality is reported per frame (inlier count, RMS residual, arc coverage in
 * degrees) so the smoothing pass can drop frames the fit could not constrain
 * rather than let them jerk the framing.
 *
 *   -r="...tl-centres.js,<timelapse.json>,<out.json>,<logPath>"
 */

const DT_INT32 = 5;
const DT_UINT16ARRAY = 23;
const DT_UINT8ARRAY = 25;
const HEADER_BYTES = 178;

// Row scan step, in half-resolution pixels.
const SCAN_STEP = 2;

// Limb threshold as a fraction of the frame's bright level. Low on purpose.
const EDGE_FRACTION = 0.10;

// Kasa iterations and inlier tolerance schedule, in px.
const FIT_ITERS = 12;
const TOL_EARLY = 40;
const TOL_LATE = 14;

// Ring search (totality) parameters.
const RING_SAMPLES = 360;
const COARSE_STEP = 8;
const REFINE_STEPS = [ 4, 2, 1 ];
const TRACK_WINDOW = 70;

// Sun/Moon apparent radius ratio on 2024-04-08: 1919" against 2010".
const SUN_OVER_MOON = 0.9547;

// Radius scan bracket for the bootstrap ring search, half-resolution px.
const R_MIN = 180;
const R_MAX = 340;

// A Kasa fit this poorly constrained is not trusted, and the frame is
// re-acquired with a global ring search before refitting.
const REACQUIRE_ARC = 100;
const REACQUIRE_N = 60;

function main()
{
   let cfgPath = jsArguments[0];
   let outPath = jsArguments[1];
   let logPath = jsArguments[2];

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
      let cfg = JSON.parse( File.readFile( cfgPath ).utf8ToString().replace( /^\uFEFF/, "" ) );
      let frames = cfg.frames;
      log( "centres: " + frames.length + " frames" );

      let cur = null, curPath = "";
      let W = 0, H = 0, w = 0, h = 0, frameBytes = 0, maxv = 65535;
      let G = null;

      let rSun = 0, rMoon = 0;
      let last = null, lastFile = "";
      let results = [];

      for ( let k = 0; k < frames.length; ++k )
      {
         let fr = frames[k];
         if ( fr.src != curPath )
         {
            if ( cur ) cur.close();
            cur = new File;
            cur.openForReading( fr.src );
            cur.position = 18;
            let a = cur.read( DT_INT32, 6 );
            W = a[2]; H = a[3];
            maxv = a[4] == 16 ? 65535 : 255;
            frameBytes = W*H*(a[4] > 8 ? 2 : 1);
            w = W >> 1; h = H >> 1;
            G = new Float32Array( w*h );
            curPath = fr.src;
            // The mount was nudged between captures, so a track must not be
            // carried across a file boundary.
            if ( fr.file != lastFile ) { last = null; lastFile = fr.file; }
            log( "  open " + fr.file );
         }

         cur.position = HEADER_BYTES + fr.index*frameBytes;
         let s = cur.read( maxv == 65535 ? DT_UINT16ARRAY : DT_UINT8ARRAY, W*H );
         let inv = 1/maxv;
         for ( let y = 0; y < h; ++y )
         {
            let r0 = 2*y*W, r1 = r0 + W, o = y*w;
            for ( let x = 0; x < w; ++x )
            {
               let c = 2*x;
               G[o + x] = (s[r0 + c + 1] + s[r1 + c])*0.5*inv;
            }
         }

         let res;
         if ( fr.state == "unfiltered" )
         {
            if ( !(rMoon > 0) )
            {
               rMoon = (rSun > 0) ? Math.round( rSun/SUN_OVER_MOON ) : 288;
               log( "  moon limb radius " + rMoon + " px (half-res)" );
            }
            // Seed with the dark-disc-inside-bright-ring matched filter used by
            // corona-combine.js. A bare gradient ring search is not reliable
            // here: during totality the corona covers most of the frame and
            // there are plenty of circular-ish edges in it, and the search
            // wandered by more than a thousand pixels between adjacent frames.
            // Matching the actual structure - dark inside, bright around - has no
            // such ambiguity.
            let grad = logGradient( G, w, h );
            let seed = seedDisc( G, w, h, rMoon );
            let c = ringSearch( grad, w, h, rMoon, seed, TRACK_WINDOW );
            last = c;
            res = { cx: c.cx, cy: c.cy, r: rMoon, n: 0, rms: 0, arc: 360,
                    method: "ring" };
         }
         else
         {
            let pts = limbPoints( G, w, h );
            if ( !(rSun > 0) )
            {
               // Bootstrap on the first filtered frame, which is the least
               // eclipsed one available: a wide clamp seeded from the bounding
               // box. That seed is only good while the disc is nearly full - it
               // sits on the crescent later on - which is exactly why the radius
               // is measured once, here, and then held.
               let wide = kasaFit( pts.pts, pts.bcx, pts.bcy,
                                   Math.max( pts.bboxR, 60 ), 60, Math.min( w, h )/2 );
               if ( wide.arc < 120 )
               {
                  // Not a full enough disc to trust; fall back to a radius scan
                  // with the seedless ring search.
                  let grad = logGradient( G, w, h );
                  let best = { score: -Infinity };
                  for ( let R = R_MIN; R <= R_MAX; R += 4 )
                  {
                     let c = ringSearch( grad, w, h, R, null, null );
                     if ( c.score > best.score )
                        best = { cx: c.cx, cy: c.cy, r: R, score: c.score };
                  }
                  wide = kasaFit( pts.pts, best.cx, best.cy, best.r,
                                  0.85*best.r, 1.15*best.r );
               }
               rSun = wide.r;
               last = { cx: wide.cx, cy: wide.cy, score: 1 };
               log( "  sun limb radius measured at " + rSun.toFixed( 1 )
                    + " px (half-res) = " + (rSun*2).toFixed( 0 ) + " px full"
                    + "  [arc " + wide.arc + " deg, rms " + wide.rms.toFixed( 2 ) + "]" );
            }
            let seedX = last ? last.cx : pts.bcx;
            let seedY = last ? last.cy : pts.bcy;
            // Two passes: the ray march needs a seed inside the disc, and its own
            // answer is a better seed than the one it started from.
            let f = null;
            for ( let pass = 0; pass < 2; ++pass )
            {
               let lp = sunLimbPoints( G, w, h, seedX, seedY, rSun );
               if ( lp.length < 20 )
                  break;
               f = fixedRadiusFit( lp, seedX, seedY, rSun );
               seedX = f.cx; seedY = f.cy;
            }
            if ( f === null )
               f = kasaFit( pts.pts, seedX, seedY, rSun, 0.94*rSun, 1.06*rSun );
            // A thin crescent constrains almost nothing from a stale seed. Rather
            // than emit a confident wrong centre, re-acquire globally and refit.
            if ( f.arc < REACQUIRE_ARC || f.n < REACQUIRE_N )
            {
               let grad = logGradient( G, w, h );
               let c = ringSearch( grad, w, h, rSun, null, null );
               let lp = sunLimbPoints( G, w, h, c.cx, c.cy, rSun );
               if ( lp.length >= 20 )
               {
                  let f2 = fixedRadiusFit( lp, c.cx, c.cy, rSun );
                  if ( f2.arc > f.arc || f2.n > f.n )
                     f = f2;
               }
            }
            last = { cx: f.cx, cy: f.cy, score: 1 };
            res = { cx: f.cx, cy: f.cy, r: f.r, n: f.n, rms: f.rms, arc: f.arc,
                    method: "kasa" };
         }

         results.push( { i: k, file: fr.file, index: fr.index, state: fr.state,
                         cx: res.cx, cy: res.cy, r: res.r,
                         n: res.n, rms: res.rms, arc: res.arc, method: res.method } );

         if ( (k + 1) % 200 == 0 )
            log( "  " + (k + 1) + "/" + frames.length + " ("
                 + ((Date.now() - t0)/1000/(k + 1)).toFixed( 2 ) + " s/frame)" );
      }
      if ( cur ) cur.close();

      let f = new File;
      f.createForWriting( outPath );
      f.outTextLn( JSON.stringify( { width: w, height: h,
                                     rSun: rSun, rMoon: rMoon,
                                     centres: results } ) );
      f.close();
      log( "  wrote " + outPath + " in " + ((Date.now() - t0)/60000).toFixed( 1 ) + " min" );
      log( "=== CENTRES OK ===" );
   }
   catch ( e )
   {
      log( "*** CENTRES FAILED: " + e.toString() );
      if ( e.stack )
         log( String( e.stack ) );
   }
   logFile.close();
}

/*
 * Sun-limb points only: along each ray outward from a seed, the OUTERMOST
 * falling edge in log brightness.
 *
 * This replaces a row scan that took the first and last bright pixel on each
 * line. That fed the circle fit both limbs at once - the Sun's where it is
 * uncovered and the Moon's where it is not - and the radius prior does not sort
 * them out, because on a thin crescent the two arcs are only a few pixels apart
 * and both sit inside the inlier tolerance. The resulting bias depends on the
 * crescent's geometry, so it changed from capture to capture: measured in the
 * rendered frames, the Sun sat up to 76 px off centre, and the offset jumped
 * every time a new capture began. That is what made the video lurch.
 *
 * Marching outward fixes it by construction. Past the Sun's limb there is only
 * sky, so the last strong falling edge on any ray is the photosphere's edge and
 * nothing else. Rays that cross only the Moon-covered part find no edge and
 * simply contribute nothing. Scoring the LOG gradient rather than a brightness
 * threshold also avoids the classic radius-low bias, where a soft limb ramps
 * over several pixels and a level crossing lands inside the true edge.
 */
function sunLimbPoints( g, w, h, cx, cy, R )
{
   const NB = 360;
   const TINY = 1e-6;
   const D = 2;
   let pts = [];
   let rLo = Math.max( 4, Math.floor( 0.55*R ) ), rHi = Math.ceil( 1.45*R );
   for ( let k = 0; k < NB; ++k )
   {
      let a = 2*Math.PI*k/NB;
      let ca = Math.cos( a ), sa = Math.sin( a );
      let bestR = -1, best = 0;
      for ( let r = rLo; r <= rHi; ++r )
      {
         let xi = Math.round( cx + ca*(r - D) ), yi = Math.round( cy + sa*(r - D) );
         let xo = Math.round( cx + ca*(r + D) ), yo = Math.round( cy + sa*(r + D) );
         if ( xi < 0 || yi < 0 || xi >= w || yi >= h ) continue;
         if ( xo < 0 || yo < 0 || xo >= w || yo >= h ) break;
         // Falling edge going outward, in log space.
         let drop = Math.log( g[yi*w + xi] + TINY ) - Math.log( g[yo*w + xo] + TINY );
         if ( drop >= best ) { best = drop; bestR = r; }
      }
      // A real limb is a large multiplicative drop; glow gradients are gentle.
      if ( bestR > 0 && best > 0.9 )
         pts.push( { x: cx + ca*bestR, y: cy + sa*bestR } );
   }
   return pts;
}

/*
 * First and last bright pixel on each row - used only to bootstrap the radius on
 * the least eclipsed frame, where the disc is nearly full and both boundaries
 * are essentially the same circle.
 */
function limbPoints( g, w, h )
{
   let hi = 0;
   for ( let i = 0, n = w*h; i < n; ++i )
      if ( g[i] > hi ) hi = g[i];
   let thr = hi*EDGE_FRACTION;

   let pts = [];
   let bx0 = w, bx1 = 0, by0 = h, by1 = 0;
   for ( let y = 0; y < h; y += SCAN_STEP )
   {
      let row = y*w, first = -1, last = -1;
      for ( let x = 0; x < w; x += SCAN_STEP )
         if ( g[row + x] > thr )
         {
            if ( first < 0 ) first = x;
            last = x;
         }
      if ( first >= 0 )
      {
         pts.push( { x: first, y: y }, { x: last, y: y } );
         if ( first < bx0 ) bx0 = first;
         if ( last > bx1 ) bx1 = last;
         if ( y < by0 ) by0 = y;
         if ( y > by1 ) by1 = y;
      }
   }
   return { pts: pts, bcx: (bx0 + bx1)/2, bcy: (by0 + by1)/2,
            bboxR: Math.max( bx1 - bx0, by1 - by0 )/2 };
}

/*
 * Kasa algebraic circle fit, iterated with a shrinking inlier tolerance and the
 * radius clamped to a prior. Reports inlier count, RMS residual and how many
 * degrees of arc the inliers span - a thin crescent constrains very little, and
 * the caller needs to know that rather than be handed a confident wrong centre.
 */
/*
 * Centre-only fit at a KNOWN radius.
 *
 * A three-parameter circle fit needs a decent spread of arc to pin the centre.
 * As the crescent thins the usable arc shrinks to well under half the limb, and
 * the fit becomes ill conditioned along the direction perpendicular to that arc -
 * radius and centre trade off against each other, so the centre slides. Because
 * the bias depends on the crescent's orientation, it lands differently in every
 * capture, which is exactly the 10-30 px step seen at capture boundaries.
 *
 * The Sun's angular radius does not change over 45 minutes. Fixing it removes the
 * degenerate direction entirely: each limb point then says "the centre lies R
 * inward along my own ray", and the answer is their mean. Two unknowns instead of
 * three, and no trade-off left to go wrong.
 */
function fixedRadiusFit( pts, cx0, cy0, R )
{
   let cx = cx0, cy = cy0;
   for ( let it = 0; it < 8; ++it )
   {
      let tol = (it < 3) ? TOL_EARLY : TOL_LATE;
      let sx = 0, sy = 0, n = 0;
      for ( let i = 0; i < pts.length; ++i )
      {
         let dx = pts[i].x - cx, dy = pts[i].y - cy;
         let d = Math.sqrt( dx*dx + dy*dy );
         if ( d < 1 || Math.abs( d - R ) > tol )
            continue;
         // Pull each point back along its own ray by exactly R.
         sx += pts[i].x - R*dx/d;
         sy += pts[i].y - R*dy/d;
         ++n;
      }
      if ( n < 12 )
         break;
      cx = sx/n; cy = sy/n;
   }

   let n = 0, ss = 0;
   let bins = new Uint8Array( 36 );
   for ( let i = 0; i < pts.length; ++i )
   {
      let dx = pts[i].x - cx, dy = pts[i].y - cy;
      let d = Math.sqrt( dx*dx + dy*dy ) - R;
      if ( Math.abs( d ) < 8 )
      {
         ++n;
         ss += d*d;
         bins[Math.floor( (Math.atan2( dy, dx ) + Math.PI)/(2*Math.PI)*36 ) % 36] = 1;
      }
   }
   let arc = 0;
   for ( let b = 0; b < 36; ++b )
      arc += bins[b];
   return { cx: cx, cy: cy, r: R, n: n,
            rms: Math.sqrt( ss/Math.max( n, 1 ) ), arc: arc*10 };
}

function kasaFit( pts, cx0, cy0, r0, rLo, rHi )
{
   let cx = cx0, cy = cy0, r = r0;
   for ( let it = 0; it < FIT_ITERS; ++it )
   {
      let tol = (it < 3) ? TOL_EARLY : TOL_LATE;
      let Sx = 0, Sy = 0, Sxx = 0, Syy = 0, Sxy = 0, Sxz = 0, Syz = 0, Sz = 0, n = 0;
      for ( let i = 0; i < pts.length; ++i )
      {
         let p = pts[i];
         let dx = p.x - cx, dy = p.y - cy;
         if ( Math.abs( Math.sqrt( dx*dx + dy*dy ) - r ) > tol )
            continue;
         let z = p.x*p.x + p.y*p.y;
         Sx += p.x; Sy += p.y; Sxx += p.x*p.x; Syy += p.y*p.y;
         Sxy += p.x*p.y; Sxz += p.x*z; Syz += p.y*z; Sz += z; ++n;
      }
      if ( n < 20 )
         break;
      let det = Sxx*(Syy*n - Sy*Sy) - Sxy*(Sxy*n - Sy*Sx) + Sx*(Sxy*Sy - Syy*Sx);
      if ( Math.abs( det ) < 1e-6 )
         break;
      let A = (Sxz*(Syy*n - Sy*Sy) - Sxy*(Syz*n - Sy*Sz) + Sx*(Syz*Sy - Syy*Sz))/det;
      let B = (Sxx*(Syz*n - Sy*Sz) - Sxz*(Sxy*n - Sy*Sx) + Sx*(Sxy*Sz - Syz*Sx))/det;
      let C = (Sxx*(Syy*Sz - Syz*Sy) - Sxy*(Sxy*Sz - Syz*Sx) + Sxz*(Sxy*Sy - Syy*Sx))/det;
      cx = A/2;
      cy = B/2;
      r = Math.sqrt( Math.max( 1, C + cx*cx + cy*cy ) );
      r = Math.max( rLo, Math.min( rHi, r ) );
   }

   let n = 0, ss = 0;
   let bins = new Uint8Array( 36 );
   for ( let i = 0; i < pts.length; ++i )
   {
      let p = pts[i];
      let dx = p.x - cx, dy = p.y - cy;
      let d = Math.sqrt( dx*dx + dy*dy ) - r;
      if ( Math.abs( d ) < 8 )
      {
         ++n;
         ss += d*d;
         bins[Math.floor( (Math.atan2( dy, dx ) + Math.PI)/(2*Math.PI)*36 ) % 36] = 1;
      }
   }
   let arc = 0;
   for ( let b = 0; b < 36; ++b )
      arc += bins[b];
   return { cx: cx, cy: cy, r: r, n: n,
            rms: Math.sqrt( ss/Math.max( n, 1 ) ), arc: arc*10 };
}

/*
 * Gradient magnitude of log brightness, softened near zero.
 *
 * A plain log blows up in empty sky: the values there are noise about zero, and
 * log turns a ratio of two small noisy numbers into a large number, so blank sky
 * scores higher than the real limb. That is not hypothetical - it pinned the
 * totality search against the right edge of the frame, reporting the Moon at
 * x=1918 of 1920 when it was actually at x=755.
 *
 * Adding a floor at the frame's own median before taking the log keeps the
 * response logarithmic across the corona, where the dynamic range needs it,
 * while damping anything at or below sky level.
 */
function logGradient( g, w, h )
{
   // Cheap median: a strided sample is plenty for a floor.
   let samp = [];
   for ( let i = 0, n = w*h; i < n; i += 37 )
      samp.push( g[i] );
   samp.sort( ( a, b ) => a - b );
   let floor = Math.max( samp[samp.length >> 1], 1e-7 );

   let L = new Float32Array( w*h );
   for ( let i = 0, n = w*h; i < n; ++i )
      L[i] = Math.log( g[i] + floor );
   let out = new Float32Array( w*h );
   for ( let y = 1; y < h - 1; ++y )
   {
      let row = y*w;
      for ( let x = 1; x < w - 1; ++x )
      {
         let dx = L[row + x + 1] - L[row + x - 1];
         let dy = L[row + w + x] - L[row - w + x];
         out[row + x] = Math.sqrt( dx*dx + dy*dy );
      }
   }
   return out;
}

/*
 * Locate the dark disc sitting inside a bright ring, on a decimated copy.
 * Radius is known, so only the centre is searched. Scoring the ring against the
 * interior is unambiguous in a way that edge strength alone is not.
 */
function seedDisc( g, w, h, R )
{
   const DS = 8;
   const ANG = 32;
   let sw = Math.floor( w/DS ), sh = Math.floor( h/DS );
   let small = new Float32Array( sw*sh );
   for ( let y = 0; y < sh; ++y )
      for ( let x = 0; x < sw; ++x )
      {
         let acc = 0;
         for ( let j = 0; j < DS; ++j )
         {
            let row = (y*DS + j)*w + x*DS;
            for ( let i = 0; i < DS; ++i )
               acc += g[row + i];
         }
         small[y*sw + x] = acc/(DS*DS);
      }

   let rs = R/DS;
   let rIn = rs*0.70, rOut = rs*1.15;
   let cosA = new Float64Array( ANG ), sinA = new Float64Array( ANG );
   for ( let k = 0; k < ANG; ++k )
   {
      let a = 2*Math.PI*k/ANG;
      cosA[k] = Math.cos( a ); sinA[k] = Math.sin( a );
   }

   let best = { cx: w/2, cy: h/2, score: -Infinity };
   let m = Math.ceil( rOut ) + 1;
   for ( let cy = m; cy < sh - m; ++cy )
      for ( let cx = m; cx < sw - m; ++cx )
      {
         let ring = 0, disc = 0;
         for ( let k = 0; k < ANG; ++k )
         {
            ring += small[Math.round( cy + sinA[k]*rOut )*sw + Math.round( cx + cosA[k]*rOut )];
            disc += small[Math.round( cy + sinA[k]*rIn )*sw + Math.round( cx + cosA[k]*rIn )];
         }
         let sc = (ring - disc)/ANG;
         if ( sc > best.score )
            best = { cx: (cx + 0.5)*DS, cy: (cy + 0.5)*DS, score: sc };
      }
   return best;
}

function ringScore( grad, w, h, cx, cy, R, cosT, sinT )
{
   let s = 0, n = 0;
   for ( let k = 0; k < RING_SAMPLES; ++k )
   {
      let x = Math.round( cx + cosT[k]*R );
      let y = Math.round( cy + sinT[k]*R );
      if ( x < 1 || y < 1 || x >= w - 1 || y >= h - 1 )
         continue;
      s += grad[y*w + x];
      ++n;
   }
   // The Moon's limb is a closed circle inside the frame during totality, so
   // demand almost the whole ring. Accepting half a ring is what let solutions
   // hanging off the frame edge compete at all.
   if ( n < RING_SAMPLES*0.95 )
      return -Infinity;
   return s/n;
}

function ringSearch( grad, w, h, R, seed, win )
{
   let cosT = new Float64Array( RING_SAMPLES ), sinT = new Float64Array( RING_SAMPLES );
   for ( let k = 0; k < RING_SAMPLES; ++k )
   {
      let a = 2*Math.PI*k/RING_SAMPLES;
      cosT[k] = Math.cos( a ); sinT[k] = Math.sin( a );
   }

   let x0 = 0, x1 = w, y0 = 0, y1 = h;
   if ( seed && win )
   {
      x0 = Math.max( 0, Math.round( seed.cx - win ) );
      x1 = Math.min( w, Math.round( seed.cx + win ) );
      y0 = Math.max( 0, Math.round( seed.cy - win ) );
      y1 = Math.min( h, Math.round( seed.cy + win ) );
   }

   let bx = (x0 + x1) >> 1, by = (y0 + y1) >> 1, bs = -Infinity;
   for ( let cy = y0; cy < y1; cy += COARSE_STEP )
      for ( let cx = x0; cx < x1; cx += COARSE_STEP )
      {
         let sc = ringScore( grad, w, h, cx, cy, R, cosT, sinT );
         if ( sc > bs ) { bs = sc; bx = cx; by = cy; }
      }

   for ( let i = 0; i < REFINE_STEPS.length; ++i )
   {
      let step = REFINE_STEPS[i];
      let span = (i == 0 ? COARSE_STEP : REFINE_STEPS[i - 1]);
      for ( let dy = -span; dy <= span; dy += step )
         for ( let dx = -span; dx <= span; dx += step )
         {
            let sc = ringScore( grad, w, h, bx + dx, by + dy, R, cosT, sinT );
            if ( sc > bs ) { bs = sc; bx = bx + dx; by = by + dy; }
         }
   }

   let sc = bs;
   let xm = ringScore( grad, w, h, bx - 1, by, R, cosT, sinT );
   let xp = ringScore( grad, w, h, bx + 1, by, R, cosT, sinT );
   let ym = ringScore( grad, w, h, bx, by - 1, R, cosT, sinT );
   let yp = ringScore( grad, w, h, bx, by + 1, R, cosT, sinT );
   let ddx = xm + xp - 2*sc, ddy = ym + yp - 2*sc;
   let ox = (ddx < 0) ? 0.5*(xm - xp)/ddx : 0;
   let oy = (ddy < 0) ? 0.5*(ym - yp)/ddy : 0;
   if ( !(Math.abs( ox ) <= 1) ) ox = 0;
   if ( !(Math.abs( oy ) <= 1) ) oy = 0;
   return { cx: bx + ox, cy: by + oy, score: bs };
}

main();
