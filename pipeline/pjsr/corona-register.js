#engine v8   // keep within the first few lines: PixInsight only scans a short
             // prologue for this, and a longer header comment pushes it out of
             // range, leaving the script in ES5 where let and arrow functions
             // are a load error that -r mode discards silently with exit 0.

/*
 * corona-register.js - measure the shift between exposure levels directly.
 *
 * Locating the Moon independently in each level does not work across a 60x
 * exposure range. The radius comes out fine, but the centre does not: at the
 * shortest exposure the limb is marked only by a chromospheric arc on one side,
 * and at the longest the limb is washed out by bleed from a saturated inner
 * corona - its ring score is ten times weaker than a well-behaved level's, and
 * the search happily locks onto the edge of a saturated blob instead. Fits that
 * disagree by hundreds of pixels put two Moons in the merged image.
 *
 * Comparing levels against each other avoids the problem entirely. It never has
 * to decide where the Moon is, only how far one frame has moved relative to
 * another, and for that the whole edge structure contributes rather than one
 * fitted circle.
 *
 * Both frames are reduced to the gradient magnitude of LOG brightness first.
 * That is what makes a 60x exposure difference comparable: in log space a given
 * edge has roughly the same height at every exposure, so the same features stand
 * out in both frames. Normalized cross-correlation then removes any residual
 * scale and offset.
 *
 * The search is a coarse-to-fine pyramid because the Moon travels a few hundred
 * pixels across the ladder, which is far too wide a window to brute-force at full
 * resolution.
 *
 *   -r="...corona-register.js,<config.json>,<logPath>"
 *
 * Config: { "images": [ { "path":.., "level":n, "t":secs } ], "refPath":..,
 *           "out":.. }
 */

// Pyramid: decimation factors, coarsest first, with the half-width of the search
// at each level expressed in that level's own pixels.
const PYRAMID = [ { ds: 16, win: 22 }, { ds: 4, win: 8 }, { ds: 1, win: 6 } ];

// A correlation this weak means the two frames share no usable structure.
const MIN_NCC = 0.25;

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
      let cfg = JSON.parse( File.readFile( cfgPath ).utf8ToString() );
      let specs = cfg.images;

      let refIdx = 0;
      for ( let i = 0; i < specs.length; ++i )
         if ( specs[i].path == cfg.refPath )
            refIdx = i;
      log( "reference: " + baseName( specs[refIdx].path ) );

      let W = 0, H = 0;
      let pyrs = [];
      for ( let i = 0; i < specs.length; ++i )
      {
         let img = loadLuma( specs[i].path );
         if ( i == 0 ) { W = img.W; H = img.H; }
         else if ( img.W != W || img.H != H )
            throw new Error( "geometry mismatch on " + specs[i].path );
         let grad = logGradient( img.lum, W, H );
         pyrs.push( buildPyramid( grad, W, H ) );
         log( "  prepared " + baseName( specs[i].path ) );
      }

      let out = [];
      for ( let i = 0; i < specs.length; ++i )
      {
         if ( i == refIdx )
         {
            out.push( { path: specs[i].path, level: specs[i].level, t: specs[i].t,
                        dx: 0, dy: 0, ncc: 1 } );
            continue;
         }
         let m = matchPyramid( pyrs[i], pyrs[refIdx], W, H );
         let flag = (m.ncc < MIN_NCC) ? "  *** WEAK" : "";
         log( "  L" + specs[i].level + " " + baseName( specs[i].path )
              + " -> shift (" + m.dx.toFixed( 2 ) + ", " + m.dy.toFixed( 2 )
              + ") ncc=" + m.ncc.toFixed( 4 ) + flag );
         out.push( { path: specs[i].path, level: specs[i].level, t: specs[i].t,
                     dx: m.dx, dy: m.dy, ncc: m.ncc } );
      }

      let f = new File;
      f.createForWriting( cfg.out );
      f.outTextLn( JSON.stringify( { refPath: cfg.refPath, shifts: out } ) );
      f.close();
      log( "  wrote " + cfg.out + " in " + ((Date.now() - t0)/1000).toFixed( 0 ) + " s" );
      log( "=== REGISTER OK ===" );
   }
   catch ( e )
   {
      log( "*** REGISTER FAILED: " + e.toString() );
      if ( e.stack )
         log( String( e.stack ) );
   }
   logFile.close();
}

function baseName( p )
{
   return p.substring( p.lastIndexOf( "/" ) + 1 );
}

function loadLuma( path )
{
   let wins = ImageWindow.open( path );
   let win = wins[0];
   for ( let i = 1; i < wins.length; ++i )
      wins[i].forceClose();
   let img = win.mainView.image;
   let W = img.width, H = img.height;
   let lum = new Float32Array( W*H );
   img.getSamples( lum, new Rect( 0, 0, W, H ), img.numberOfChannels > 1 ? 1 : 0 );
   win.forceClose();
   return { lum: lum, W: W, H: H };
}

function logGradient( g, W, H )
{
   const TINY = 1e-9;
   let L = new Float32Array( W*H );
   for ( let i = 0, n = W*H; i < n; ++i )
      L[i] = Math.log( Math.max( g[i], TINY ) );
   let out = new Float32Array( W*H );
   for ( let y = 1; y < H - 1; ++y )
   {
      let row = y*W;
      for ( let x = 1; x < W - 1; ++x )
      {
         let dx = L[row + x + 1] - L[row + x - 1];
         let dy = L[row + W + x] - L[row - W + x];
         out[row + x] = Math.sqrt( dx*dx + dy*dy );
      }
   }
   return out;
}

function buildPyramid( full, W, H )
{
   let levels = {};
   for ( let i = 0; i < PYRAMID.length; ++i )
   {
      let ds = PYRAMID[i].ds;
      levels[ds] = (ds == 1) ? { img: full, w: W, h: H } : decimate( full, W, H, ds );
   }
   return levels;
}

function decimate( img, W, H, k )
{
   let w = Math.floor( W/k ), h = Math.floor( H/k );
   let o = new Float32Array( w*h );
   let inv = 1/(k*k);
   for ( let y = 0; y < h; ++y )
      for ( let x = 0; x < w; ++x )
      {
         let s = 0;
         for ( let j = 0; j < k; ++j )
         {
            let row = (y*k + j)*W + x*k;
            for ( let i = 0; i < k; ++i )
               s += img[row + i];
         }
         o[y*w + x] = s*inv;
      }
   return { img: o, w: w, h: h };
}

/* Normalized cross-correlation of b shifted by (dx, dy) against a. */
function ncc( a, b, w, h, dx, dy )
{
   let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, n = 0;
   let y0 = Math.max( 0, -dy ), y1 = Math.min( h, h - dy );
   let x0 = Math.max( 0, -dx ), x1 = Math.min( w, w - dx );
   for ( let y = y0; y < y1; ++y )
   {
      let ra = y*w, rb = (y + dy)*w + dx;
      for ( let x = x0; x < x1; ++x )
      {
         let u = a[ra + x], v = b[rb + x];
         sa += u; sb += v; saa += u*u; sbb += v*v; sab += u*v; ++n;
      }
   }
   if ( n < 1000 )
      return -2;
   let ma = sa/n, mb = sb/n;
   let ca = saa - n*ma*ma, cb = sbb - n*mb*mb;
   if ( ca <= 0 || cb <= 0 )
      return -2;
   return (sab - n*ma*mb)/Math.sqrt( ca*cb );
}

/*
 * Coarse-to-fine match. Each pyramid step searches a small window around the
 * previous step's answer, scaled into its own resolution, so the total window is
 * hundreds of pixels while the work stays bounded.
 */
function matchPyramid( pa, pb, W, H )
{
   let bx = 0, by = 0, bs = -2;
   for ( let i = 0; i < PYRAMID.length; ++i )
   {
      let ds = PYRAMID[i].ds, win = PYRAMID[i].win;
      let A = pa[ds], B = pb[ds];
      let cx = Math.round( bx/ds ), cy = Math.round( by/ds );
      let best = { dx: cx, dy: cy, s: -2 };
      for ( let dy = cy - win; dy <= cy + win; ++dy )
         for ( let dx = cx - win; dx <= cx + win; ++dx )
         {
            let s = ncc( B.img, A.img, A.w, A.h, dx, dy );
            if ( s > best.s ) best = { dx: dx, dy: dy, s: s };
         }
      bx = best.dx*ds; by = best.dy*ds; bs = best.s;
   }

   // Sub-pixel parabola at full resolution.
   let A = pa[1], B = pb[1];
   let c = ncc( B.img, A.img, W, H, bx, by );
   let xm = ncc( B.img, A.img, W, H, bx - 1, by );
   let xp = ncc( B.img, A.img, W, H, bx + 1, by );
   let ym = ncc( B.img, A.img, W, H, bx, by - 1 );
   let yp = ncc( B.img, A.img, W, H, bx, by + 1 );
   let ddx = xm + xp - 2*c, ddy = ym + yp - 2*c;
   let ox = (ddx < 0) ? 0.5*(xm - xp)/ddx : 0;
   let oy = (ddy < 0) ? 0.5*(ym - yp)/ddy : 0;
   if ( !(Math.abs( ox ) <= 1) ) ox = 0;
   if ( !(Math.abs( oy ) <= 1) ) oy = 0;

   // Returned as the shift to APPLY to this image to bring it onto the reference.
   return { dx: -(bx + ox), dy: -(by + oy), ncc: c };
}

main();
