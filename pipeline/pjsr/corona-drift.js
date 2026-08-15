#engine v8   // must sit in the first few lines: PixInsight only scans a
             // short prologue for it, and a header comment long enough to
             // push it past ~line 20 leaves the script in ES5, where let
             // and arrow functions are a load error that -r mode discards
             // silently with exit 0 and no log.
/*
 * corona-drift.js — measure how fast the Moon slides across the corona.
 *
 * The Moon and the corona are not the same reference frame. The corona is fixed
 * to the Sun; the Moon crosses it at roughly its own synodic rate. Over one 60 s
 * capture that is a handful of pixels, but across the whole exposure ladder it is
 * tens of pixels — enough to double the streamers in an HDR merge, or to double
 * the limb, depending on which feature the registration happened to lock onto.
 *
 * Rather than trust an ephemeris and an assumed image scale, this measures the
 * rate from the data: take two stacks of the SAME exposure level at different
 * times, put them in a common Moon-centred frame, mask out everything but the
 * outer corona, and cross-correlate. Whatever shift remains is the Moon-vs-Sun
 * slip over the known interval.
 *
 * The correlation runs on the radially flattened image. Without that, the profile
 * itself dominates the correlation and every candidate shift near zero scores
 * about the same, because a smooth radial gradient correlates with itself.
 *
 *   -r="...corona-drift.js,<config.json>,<logPath>"
 *
 * Config: { "a": {"path":...,"cx":...,"cy":...,"r":...,"t":secs},
 *           "b": {...}, "out": "drift.json" }
 */


// Annulus used for the correlation, in units of the Moon's radius. The inner
// bound clears the limb and the chromosphere. The outer bound is deliberately
// close in: a first attempt reaching out to 3.2 R measured a drift three times
// the physical ceiling, because that far out the corona is faint and the
// correlation locked onto the sensor's own flat-field signature instead, which
// does not move between frames and so reported the Moon's full frame motion as
// if it were differential. Structure this close in is unambiguously corona.
const R_INNER = 1.10;
const R_OUTER = 2.00;

// Physical ceiling on the differential rate. The Moon moves relative to the Sun
// at the synodic rate; converting through the measured Moon radius turns that
// into pixels per second without needing to know the focal length. Anything
// above this is a failed correlation, not a fast Moon - the topocentric rate
// during totality is lower still, so the margin is generous.
const SYNODIC_ARCSEC_PER_S = 0.51;
const MOON_RADIUS_ARCSEC = 990.0;
const DRIFT_SAFETY = 1.4;

// Coarse search is done on an 8x-decimated image, so this window covers
// +/-(SEARCH_COARSE*8) full-resolution pixels.
const SEARCH_COARSE = 14;
const DECIM = 8;
const SEARCH_FINE = 6;

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
      let dt = cfg.b.t - cfg.a.t;
      if ( !(Math.abs( dt ) > 1 ) )
         throw new Error( "need a meaningful time baseline, got dt=" + dt + "s" );

      let A = ecLoadPlane( cfg.a, log );
      let B = ecLoadPlane( cfg.b, log );
      if ( A.W != B.W || A.H != B.H )
         throw new Error( "geometry mismatch between the two stacks" );
      let W = A.W, H = A.H;

      // Common Moon-centred frame: shift B so its Moon sits on A's Moon. Any
      // residual corona shift after this is the differential motion. The offset
      // comes from corona-register.js when available, since differencing two
      // independent Moon fits is exactly what proved unreliable.
      let mx, my;
      if ( cfg.b.dx !== undefined && cfg.a.dx !== undefined )
      {
         mx = cfg.b.dx - cfg.a.dx;
         my = cfg.b.dy - cfg.a.dy;
      }
      else
      {
         mx = cfg.a.cx - cfg.b.cx;
         my = cfg.a.cy - cfg.b.cy;
      }
      let bAligned = translateImage( B.lum, W, H, mx, my );
      log( "  moon-registered B onto A: (" + mx.toFixed( 2 ) + ", "
           + my.toFixed( 2 ) + ") px" );

      let rIn = cfg.a.r*R_INNER, rOut = cfg.a.r*R_OUTER;
      let fa = flattenAnnulus( A.lum, W, H, cfg.a.cx, cfg.a.cy, rIn, rOut );
      let fb = flattenAnnulus( bAligned, W, H, cfg.a.cx, cfg.a.cy, rIn, rOut );

      // Coarse pass on a decimated grid, then refine at full resolution.
      let dA = decimate( fa.img, fa.mask, W, H, DECIM );
      let dB = decimate( fb.img, fb.mask, W, H, DECIM );
      let coarse = search( dA, dB, dA.w, dA.h, SEARCH_COARSE );
      log( "  coarse best (" + (coarse.dx*DECIM) + ", " + (coarse.dy*DECIM)
           + ") ncc=" + coarse.score.toFixed( 5 ) );

      let fine = searchAround( fa, fb, W, H,
                               coarse.dx*DECIM, coarse.dy*DECIM, SEARCH_FINE );
      log( "  refined best (" + fine.dx.toFixed( 2 ) + ", " + fine.dy.toFixed( 2 )
           + ") ncc=" + fine.score.toFixed( 5 ) );

      // fine.* is how far B's corona sits from A's after Moon registration, so
      // the corona moves by -fine per dt in the Moon frame.
      let vx = -fine.dx/dt, vy = -fine.dy/dt;
      let speed = Math.sqrt( vx*vx + vy*vy );
      log( "  baseline " + dt.toFixed( 1 ) + " s" );
      log( "  differential drift " + speed.toFixed( 4 ) + " px/s  ("
           + vx.toFixed( 4 ) + ", " + vy.toFixed( 4 ) + ")" );
      log( "  => " + (speed*60).toFixed( 1 ) + " px per 60 s capture" );

      // Physics gate. A measured rate above the synodic ceiling means the
      // correlation found something that is not corona; report it and hand back
      // zero so the caller falls back to rigid registration rather than shifting
      // every level by a fabricated amount.
      let ceiling = DRIFT_SAFETY*SYNODIC_ARCSEC_PER_S*cfg.a.r/MOON_RADIUS_ARCSEC;
      let accepted = speed <= ceiling;
      log( "  ceiling " + ceiling.toFixed( 4 ) + " px/s (moon r=" + cfg.a.r.toFixed( 1 )
           + " px) -> " + (accepted ? "accepted" : "REJECTED") );
      if ( !accepted )
      {
         log( "  *** measured rate exceeds what the Moon can do; reporting zero drift" );
         vx = 0; vy = 0;
      }

      let out = new File;
      out.createForWriting( cfg.out );
      out.outTextLn( JSON.stringify( {
         dtSeconds: dt,
         shiftPx: { dx: fine.dx, dy: fine.dy },
         driftPxPerSec: { x: vx, y: vy, speed: Math.sqrt( vx*vx + vy*vy ) },
         measuredSpeed: speed,
         ceiling: ceiling,
         accepted: accepted,
         ncc: fine.score,
         annulus: { rInner: rIn, rOuter: rOut },
      } ) );
      out.close();

      log( "  wrote " + cfg.out + " in " + ((Date.now() - t0)/1000).toFixed( 0 ) + " s" );
      log( "=== DRIFT OK ===" );
   }
   catch ( e )
   {
      log( "*** DRIFT FAILED: " + e.toString() );
      if ( e.stack )
         log( String( e.stack ) );
   }
   logFile.close();
}

function ecLoadPlane( spec, log )
{
   let wins = ImageWindow.open( spec.path );
   let win = wins[0];
   for ( let i = 1; i < wins.length; ++i )
      wins[i].forceClose();
   let img = win.mainView.image;
   let W = img.width, H = img.height;
   let lum = new Float32Array( W*H );
   img.getSamples( lum, new Rect( 0, 0, W, H ), img.numberOfChannels > 1 ? 1 : 0 );
   win.forceClose();
   log( "  loaded " + spec.path.substring( spec.path.lastIndexOf( "/" ) + 1 ) + " " + W + "x" + H );
   return { lum: lum, W: W, H: H };
}

/*
 * Divide out the radial median profile inside the annulus and return the result
 * plus a validity mask. Flattening is what gives the correlation something to
 * lock onto: the streamers, not the falloff.
 */
function flattenAnnulus( g, W, H, cx, cy, rIn, rOut )
{
   let n = W*H;
   let rmaxI = Math.ceil( rOut );
   let sum = new Float64Array( rmaxI + 1 );
   let cnt = new Int32Array( rmaxI + 1 );
   for ( let y = 0; y < H; ++y )
   {
      let dy = y - cy, row = y*W;
      for ( let x = 0; x < W; ++x )
      {
         let dx = x - cx;
         let d = Math.sqrt( dx*dx + dy*dy );
         if ( d < rIn || d > rOut ) continue;
         let r = Math.round( d );
         sum[r] += g[row + x];
         cnt[r]++;
      }
   }
   let prof = new Float64Array( rmaxI + 1 );
   for ( let r = 0; r <= rmaxI; ++r )
      prof[r] = cnt[r] > 0 ? sum[r]/cnt[r] : 0;

   let img = new Float32Array( n );
   let mask = new Uint8Array( n );
   for ( let y = 0; y < H; ++y )
   {
      let dy = y - cy, row = y*W;
      for ( let x = 0; x < W; ++x )
      {
         let dx = x - cx;
         let d = Math.sqrt( dx*dx + dy*dy );
         if ( d < rIn || d > rOut ) continue;
         let r = Math.round( d );
         let p = prof[r];
         if ( !(p > 0) ) continue;
         img[row + x] = g[row + x]/p - 1;   // fractional excess over the profile
         mask[row + x] = 1;
      }
   }
   return { img: img, mask: mask };
}

function decimate( img, mask, W, H, k )
{
   let w = Math.floor( W/k ), h = Math.floor( H/k );
   let oi = new Float32Array( w*h );
   let om = new Uint8Array( w*h );
   for ( let y = 0; y < h; ++y )
      for ( let x = 0; x < w; ++x )
      {
         let s = 0, c = 0;
         for ( let j = 0; j < k; ++j )
         {
            let row = (y*k + j)*W + x*k;
            for ( let i = 0; i < k; ++i )
               if ( mask[row + i] ) { s += img[row + i]; c++; }
         }
         if ( c > (k*k >> 1) ) { oi[y*w + x] = s/c; om[y*w + x] = 1; }
      }
   return { img: oi, mask: om, w: w, h: h };
}

/* Normalized cross-correlation over integer shifts, masked. */
function ncc( a, b, W, H, dx, dy )
{
   let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, n = 0;
   for ( let y = 0; y < H; ++y )
   {
      let sy = y + dy;
      if ( sy < 0 || sy >= H ) continue;
      let ra = y*W, rb = sy*W;
      for ( let x = 0; x < W; ++x )
      {
         let sx = x + dx;
         if ( sx < 0 || sx >= W ) continue;
         if ( !a.mask[ra + x] || !b.mask[rb + sx] ) continue;
         let u = a.img[ra + x], v = b.img[rb + sx];
         sa += u; sb += v; saa += u*u; sbb += v*v; sab += u*v; n++;
      }
   }
   if ( n < 500 ) return -2;
   let ma = sa/n, mb = sb/n;
   let ca = saa - n*ma*ma, cb = sbb - n*mb*mb;
   if ( ca <= 0 || cb <= 0 ) return -2;
   return (sab - n*ma*mb)/Math.sqrt( ca*cb );
}

function search( a, b, W, H, win )
{
   let best = { dx: 0, dy: 0, score: -2 };
   for ( let dy = -win; dy <= win; ++dy )
      for ( let dx = -win; dx <= win; ++dx )
      {
         let s = ncc( a, b, W, H, dx, dy );
         if ( s > best.score ) best = { dx: dx, dy: dy, score: s };
      }
   return best;
}

function searchAround( a, b, W, H, cx, cy, win )
{
   let best = { dx: cx, dy: cy, score: -2 };
   for ( let dy = cy - win; dy <= cy + win; ++dy )
      for ( let dx = cx - win; dx <= cx + win; ++dx )
      {
         let s = ncc( a, b, W, H, dx, dy );
         if ( s > best.score ) best = { dx: dx, dy: dy, score: s };
      }
   return best;
}

function translateImage( src, W, H, dx, dy )
{
   let dst = new Float32Array( W*H );
   for ( let y = 0; y < H; ++y )
   {
      let sy = y - dy;
      let y0 = Math.floor( sy ), fy = sy - y0, y1 = y0 + 1;
      if ( y1 < 0 || y0 >= H ) continue;
      if ( y0 < 0 ) { y0 = 0; fy = 0; }
      if ( y1 >= H ) y1 = H - 1;
      let r0 = y0*W, r1 = y1*W, drow = y*W;
      for ( let x = 0; x < W; ++x )
      {
         let sx = x - dx;
         let x0 = Math.floor( sx ), fx = sx - x0, x1 = x0 + 1;
         if ( x1 < 0 || x0 >= W ) continue;
         if ( x0 < 0 ) { x0 = 0; fx = 0; }
         if ( x1 >= W ) x1 = W - 1;
         let u = src[r0 + x0] + (src[r0 + x1] - src[r0 + x0])*fx;
         let v = src[r1 + x0] + (src[r1 + x1] - src[r1 + x0])*fx;
         dst[drow + x] = u + (v - u)*fy;
      }
   }
   return dst;
}

main();
