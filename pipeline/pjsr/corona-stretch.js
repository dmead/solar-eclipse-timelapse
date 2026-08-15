#engine v8   // keep within the first few lines: PixInsight only scans a short
             // prologue for this, and a longer header comment pushes it out of
             // range, leaving the script in ES5 where let and arrow functions
             // are a load error that -r mode discards silently with exit 0.

/*
 * corona-stretch.js - make the merged corona presentable.
 *
 * INPUT IS corona_hdr.xisf, THE UNFLATTENED MERGE. Running this on corona_flat
 * instead produces a false-colour mess: the flatten has already divided out the
 * radial falloff, so the disc interior and the outer field sit at similar levels,
 * and asinh then amplifies all of it - including the residual inside the Moon -
 * into saturation. The flatten and this stretch are two answers to the same
 * problem and must not be stacked. Use the flatten when you want a
 * structure-only map; use this for a picture.
 *
 * corona_hdr.xisf is geometrically correct but tonally useless: the
 * chromosphere sits near 1.0 and the outer corona near 0.001, seven stops down,
 * and any gamma that lifts the outer corona washes the inner corona to white.
 *
 * Three steps, in order.
 *
 * 1. SKY PEDESTAL. Totality sky is not black - it is dusk. That pedestal adds to
 *    every pixel, so it flattens contrast everywhere and, being additive, it
 *    breaks the stretch below: asinh applied to signal+pedestal compresses the
 *    signal by whatever the pedestal already used up. Measured as a low
 *    percentile in an annulus outside the corona and subtracted per channel,
 *    which also removes most of the colour cast.
 *
 * 2. ASINH STRETCH. Logarithmic for large values, linear for small ones, so the
 *    faint outer corona is lifted hard while the inner corona is compressed
 *    gently rather than clipped. Unlike a per-channel gamma it applies one gain
 *    derived from luminance to all three channels, so colour ratios survive -
 *    a per-channel curve would desaturate the bright inner region toward white.
 *
 * 3. LOCAL CONTRAST. Streamers are low-contrast structure on a smooth gradient.
 *    Subtracting a blurred copy and adding back a fraction of the difference
 *    raises them without touching overall brightness.
 *
 *   -r="...corona-stretch.js,<in.xisf>,<out.xisf>,<moon.json>,<logPath>"
 */

// Annulus (in Moon radii) used to measure the sky pedestal. Outside any real
// corona but still inside the frame.
const SKY_R_INNER = 3.2;
const SKY_R_OUTER = 3.9;
const SKY_PERCENTILE = 0.10;

// asinh strength. Larger lifts faint signal harder.
const ASINH_BETA = 120.0;

// Target for the inner corona just outside the limb, after stretching.
const INNER_TARGET = 0.72;

/*
 * Local contrast, at two scales.
 *
 * Streamers are broad, low-contrast structure sitting on a steep radial
 * gradient. A single small-radius unsharp mask sharpens noise and edges without
 * touching them; the large radius is the one that actually separates a streamer
 * from its background. The small radius is kept, gently, for the fine polar
 * plumes near the limb.
 *
 * Doing it this way rather than by radial flattening is deliberate: flattening
 * before an asinh stretch was tried twice and both times produced a false-colour
 * image, because it lifts the outer field to the same level as the inner corona
 * and the stretch then saturates everything. Local contrast raises structure
 * without changing the overall brightness envelope, so the picture stays a
 * corona.
 */
const LC_RADIUS = 20;
const LC_AMOUNT = 0.30;
const LC2_RADIUS = 110;
const LC2_AMOUNT = 0.85;

function main()
{
   let inPath = jsArguments[0];
   let outPath = jsArguments[1];
   let moonPath = jsArguments[2];
   let logPath = jsArguments[3];

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
      let moon = JSON.parse( File.readFile( moonPath ).utf8ToString() );

      let wins = ImageWindow.open( inPath );
      let win = wins[0];
      for ( let i = 1; i < wins.length; ++i )
         wins[i].forceClose();
      let img = win.mainView.image;
      let W = img.width, H = img.height, n = W*H;
      let ch = [];
      for ( let c = 0; c < 3; ++c )
      {
         let a = new Float32Array( n );
         img.getSamples( a, new Rect( 0, 0, W, H ), c );
         ch.push( a );
      }
      win.forceClose();

      // The sidecar centre is in the UNCROPPED frame; corona-hdr trimmed to
      // common coverage, so shift it by the trim offset when one was recorded.
      let cx = moon.cx - (moon.trimX || 0), cy = moon.cy - (moon.trimY || 0);
      let R = moon.radius;
      log( "stretch " + W + "x" + H + " moon (" + cx.toFixed( 0 ) + ", "
           + cy.toFixed( 0 ) + ") r=" + R.toFixed( 0 ) );

      // ---- 1. sky pedestal ----
      let rIn = SKY_R_INNER*R, rOut = SKY_R_OUTER*R;
      let rIn2 = rIn*rIn, rOut2 = rOut*rOut;
      for ( let c = 0; c < 3; ++c )
      {
         let vals = [];
         for ( let y = 0; y < H; y += 2 )
         {
            let dy = y - cy, row = y*W;
            for ( let x = 0; x < W; x += 2 )
            {
               let dx = x - cx, d2 = dx*dx + dy*dy;
               if ( d2 >= rIn2 && d2 <= rOut2 )
                  vals.push( ch[c][row + x] );
            }
         }
         let sky = 0;
         if ( vals.length > 200 )
         {
            vals.sort( ( a, b ) => a - b );
            sky = vals[Math.floor( vals.length*SKY_PERCENTILE )];
         }
         log( "  channel " + c + " sky pedestal " + sky.toExponential( 3 )
              + " from " + vals.length + " samples" );
         for ( let p = 0; p < n; ++p )
         {
            let v = ch[c][p] - sky;
            ch[c][p] = v > 0 ? v : 0;
         }
      }

      // ---- 1b. white balance on the corona itself ----
      // The three channels carry different sky pedestals and different sensor
      // response, and subtracting unequal pedestals leaves a cast of its own -
      // the first result had a distinctly green limb. The corona is very nearly
      // white in reality, so equalising the channel medians over an annulus of
      // real corona is a defensible balance and costs nothing in structure.
      let wbLo = 1.15*R, wbHi = 2.10*R;
      let med = [];
      for ( let c = 0; c < 3; ++c )
         med.push( annulusMedianCh( ch[c], W, H, cx, cy, wbLo, wbHi ) );
      let mref = (med[0] + med[1] + med[2])/3;
      for ( let c = 0; c < 3; ++c )
      {
         if ( !(med[c] > 0) )
            continue;
         let g = mref/med[c];
         log( "  wb channel " + c + " median " + med[c].toExponential( 3 )
              + " gain " + g.toFixed( 3 ) );
         for ( let p = 0; p < n; ++p )
            ch[c][p] *= g;
      }

      // ---- 2. asinh stretch, luminance-driven ----
      // Normalize so the inner corona lands at INNER_TARGET after stretching.
      let ref = annulusMedian( ch, W, H, cx, cy, 1.05*R, 1.25*R );
      if ( !(ref > 0) )
         throw new Error( "inner corona reference measured as zero" );
      let k = asinhSolve( ref, INNER_TARGET );
      log( "  inner corona ref " + ref.toExponential( 3 ) + ", scale " + k.toFixed( 1 )
           + ", beta " + ASINH_BETA );

      let denom = Math.asinh( ASINH_BETA );
      for ( let p = 0; p < n; ++p )
      {
         let L = 0.25*ch[0][p] + 0.60*ch[1][p] + 0.15*ch[2][p];
         if ( L <= 0 )
            continue;
         // One gain from luminance, applied to all three channels: this is what
         // keeps the colour of the chromosphere instead of bleaching it.
         let s = Math.asinh( L*k*ASINH_BETA )/denom;
         let gain = s/L;
         for ( let c = 0; c < 3; ++c )
         {
            let v = ch[c][p]*gain;
            ch[c][p] = v > 1 ? 1 : v;
         }
      }

      // ---- 2b. blank the Moon ----
      // Whatever survives inside the lunar disc is scattered light and stack
      // residual, not corona. asinh lifts it to a visible grey plate and the eye
      // reads that as fog over the whole image, so it is taken to black with a
      // short feather across the limb.
      let rBlank = 0.985*R, rFeather = 0.045*R;
      for ( let y = 0; y < H; ++y )
      {
         let dy = y - cy, row = y*W;
         for ( let x = 0; x < W; ++x )
         {
            let dx = x - cx;
            let d = Math.sqrt( dx*dx + dy*dy );
            if ( d >= rBlank )
               continue;
            let f = (d > rBlank - rFeather) ? (d - (rBlank - rFeather))/rFeather : 0;
            for ( let c = 0; c < 3; ++c )
               ch[c][row + x] *= f;
         }
      }
      log( "  moon interior blanked inside r=" + rBlank.toFixed( 0 ) );

      // ---- 3. local contrast ----
      let scales = [ [ LC2_RADIUS, LC2_AMOUNT ], [ LC_RADIUS, LC_AMOUNT ] ];
      for ( let si = 0; si < scales.length; ++si )
      {
         let rad = scales[si][0], amt = scales[si][1];
         if ( !(amt > 0) )
            continue;
         for ( let c = 0; c < 3; ++c )
         {
            let blur = boxBlur( ch[c], W, H, rad );
            for ( let p = 0; p < n; ++p )
            {
               let v = ch[c][p] + amt*(ch[c][p] - blur[p]);
               ch[c][p] = v < 0 ? 0 : (v > 1 ? 1 : v);
            }
         }
         log( "  local contrast r=" + rad + " amount " + amt );
      }

      let outWin = new ImageWindow( W, H, 3, 32, true, true, "corona_final" );
      let v = outWin.mainView;
      v.beginProcess( UndoFlag.NoSwapFile );
      for ( let c = 0; c < 3; ++c )
      {
         v.image.selectedChannel = c;
         v.image.setSamples( ch[c] );
      }
      v.image.resetSelections();
      v.endProcess();
      if ( !outWin.saveAs( outPath, false, false, false, false ) )
         throw new Error( "save failed: " + outPath );
      outWin.forceClose();

      log( "  saved " + outPath + " in " + ((Date.now() - t0)/1000).toFixed( 0 ) + " s" );
      log( "=== STRETCH OK ===" );
   }
   catch ( e )
   {
      log( "*** STRETCH FAILED: " + e.toString() );
      if ( e.stack )
         log( String( e.stack ) );
   }
   logFile.close();
}

/* Median luminance in an annulus - a robust handle on "how bright is the corona
 * just outside the limb", unaffected by prominences or a stray streamer. */
function annulusMedianCh( a, W, H, cx, cy, rIn, rOut )
{
   let rIn2 = rIn*rIn, rOut2 = rOut*rOut;
   let vals = [];
   for ( let y = 0; y < H; y += 2 )
   {
      let dy = y - cy, row = y*W;
      for ( let x = 0; x < W; x += 2 )
      {
         let dx = x - cx, d2 = dx*dx + dy*dy;
         if ( d2 >= rIn2 && d2 <= rOut2 )
            vals.push( a[row + x] );
      }
   }
   if ( vals.length < 100 )
      return 0;
   vals.sort( ( u, v ) => u - v );
   return vals[vals.length >> 1];
}

function annulusMedian( ch, W, H, cx, cy, rIn, rOut )
{
   let rIn2 = rIn*rIn, rOut2 = rOut*rOut;
   let vals = [];
   for ( let y = 0; y < H; y += 2 )
   {
      let dy = y - cy, row = y*W;
      for ( let x = 0; x < W; x += 2 )
      {
         let dx = x - cx, d2 = dx*dx + dy*dy;
         if ( d2 >= rIn2 && d2 <= rOut2 )
            vals.push( 0.25*ch[0][row + x] + 0.60*ch[1][row + x] + 0.15*ch[2][row + x] );
      }
   }
   if ( vals.length < 100 )
      return 0;
   vals.sort( ( a, b ) => a - b );
   return vals[vals.length >> 1];
}

/* Scale k such that asinh(ref*k*beta)/asinh(beta) == target. */
function asinhSolve( ref, target )
{
   let denom = Math.asinh( ASINH_BETA );
   let lo = 1e-3, hi = 1e9;
   for ( let i = 0; i < 200; ++i )
   {
      let mid = Math.sqrt( lo*hi );
      if ( Math.asinh( ref*mid*ASINH_BETA )/denom < target )
         lo = mid;
      else
         hi = mid;
   }
   return Math.sqrt( lo*hi );
}

/* Separable box blur, two passes - close enough to Gaussian for local contrast
 * and far cheaper on an 8 Mpx frame. */
function boxBlur( src, W, H, r )
{
   let tmp = new Float32Array( W*H );
   let out = new Float32Array( W*H );
   for ( let pass = 0; pass < 2; ++pass )
   {
      let a = (pass === 0) ? src : out;
      // horizontal
      for ( let y = 0; y < H; ++y )
      {
         let row = y*W, sum = 0, cnt = 0;
         for ( let x = 0; x <= r && x < W; ++x ) { sum += a[row + x]; ++cnt; }
         for ( let x = 0; x < W; ++x )
         {
            tmp[row + x] = sum/cnt;
            let add = x + r + 1, sub = x - r;
            if ( add < W ) { sum += a[row + add]; ++cnt; }
            if ( sub >= 0 ) { sum -= a[row + sub]; --cnt; }
         }
      }
      // vertical
      for ( let x = 0; x < W; ++x )
      {
         let sum = 0, cnt = 0;
         for ( let y = 0; y <= r && y < H; ++y ) { sum += tmp[y*W + x]; ++cnt; }
         for ( let y = 0; y < H; ++y )
         {
            out[y*W + x] = sum/cnt;
            let add = y + r + 1, sub = y - r;
            if ( add < H ) { sum += tmp[add*W + x]; ++cnt; }
            if ( sub >= 0 ) { sum -= tmp[sub*W + x]; --cnt; }
         }
      }
   }
   return out;
}

main();
