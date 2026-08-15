#engine v8   // keep within the first few lines: PixInsight only scans a short
             // prologue for this, and a longer header comment pushes it out of
             // range, leaving the script in ES5 where let and arrow functions
             // are a load error that -r mode discards silently with exit 0.

/*
 * corona-combine.js - assemble the three per-channel stacks of one exposure
 * level into a linear colour image, and locate the Moon.
 *
 * ser-stack.js extracts one CFA channel per run, so each channel arrives as a
 * separate mono XISF. Because the Bayer extraction halves resolution and the
 * drizzle puts the factor of two back, the channels land at the sensor's native
 * 3840x2160 and combine without any resampling.
 *
 * The Moon's centre and radius go to a sidecar: every later stage needs them, for
 * registration between exposure levels and for the radial profile that makes the
 * outer corona visible.
 *
 * FINDING THE MOON
 *
 * Earlier versions cast rays outward and fitted a circle to where the brightness
 * crossed a threshold. The radius that produced was excellent - 565.8 px on every
 * level, to a tenth of a pixel - but the centre was not: tracking it through
 * totality gave apparent frame motion of 66, 3.2, 12, 2.7 and 0.82 px/s on
 * successive levels, and the Moon cannot change speed like that. A least-squares
 * circle through crossings is only as good as the crossings, and prominences, a
 * lopsided inner corona and saturation bias them asymmetrically, which moves the
 * centre while leaving the radius about right.
 *
 * This instead scores candidate centres directly: the limb is the strongest edge
 * in the frame at every exposure, so the true centre is the one whose circle of
 * radius R lies along the most edge. Scoring on the gradient of LOG brightness
 * makes that exposure-invariant - it asks where brightness multiplies fastest,
 * not where it exceeds a level - and searching the whole frame hierarchically
 * means an asymmetric feature cannot drag the answer off, it can only fail to
 * improve the score.
 *
 *   -r="...corona-combine.js,<R.xisf>,<G.xisf>,<B.xisf>,<out.xisf>,<logPath>[,<fixedRadius>]"
 */

// Ring sampling for the centre search.
const RING_SAMPLES = 360;

// Hierarchical search: start on this grid, then refine by halving.
const COARSE_STEP = 16;
const REFINE_STEPS = [ 8, 4, 2, 1 ];

// Radius search bracket around the seed, when no radius is imposed.
const R_LO_FRAC = 0.55;
const R_HI_FRAC = 1.70;
const R_SEARCH_STEPS = 24;

function main()
{
   let paths = [ jsArguments[0], jsArguments[1], jsArguments[2] ];
   let outPath = jsArguments[3];
   let logPath = jsArguments[4];
   let fixedR = (jsArguments.length > 5) ? parseFloat( jsArguments[5] ) : 0;

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
      let W = 0, H = 0;
      let planes = [];

      for ( let c = 0; c < 3; ++c )
      {
         let wins = ImageWindow.open( paths[c] );
         let win = wins[0];
         for ( let i = 1; i < wins.length; ++i )
            wins[i].forceClose();
         let img = win.mainView.image;
         if ( c == 0 )
         {
            W = img.width;
            H = img.height;
         }
         else if ( img.width != W || img.height != H )
            throw new Error( "channel geometry mismatch: " + paths[c]
                             + " is " + img.width + "x" + img.height
                             + ", expected " + W + "x" + H );
         let a = new Float32Array( W*H );
         img.getSamples( a, new Rect( 0, 0, W, H ), 0 );
         planes.push( a );
         win.forceClose();
      }
      log( "combine " + W + "x" + H );

      let moon = measureMoon( planes[1], W, H, log, fixedR );

      let outWin = new ImageWindow( W, H, 3, 32, true, true, "corona" );
      let v = outWin.mainView;
      v.beginProcess( UndoFlag.NoSwapFile );
      for ( let c = 0; c < 3; ++c )
      {
         v.image.selectedChannel = c;
         v.image.setSamples( planes[c] );
      }
      v.image.resetSelections();
      v.endProcess();

      if ( !outWin.saveAs( outPath, false, false, false, false ) )
         throw new Error( "save failed: " + outPath );
      outWin.forceClose();

      let side = new File;
      side.createForWriting( outPath.replace( /\.xisf$/i, "" ) + "_moon.json" );
      side.outTextLn( JSON.stringify( {
         image: outPath, width: W, height: H,
         cx: moon.cx, cy: moon.cy, radius: moon.r, score: moon.score,
      } ) );
      side.close();

      log( "  saved " + outPath + " in " + ((Date.now() - t0)/1000).toFixed( 0 ) + " s" );
      log( "=== COMBINE OK ===" );
   }
   catch ( e )
   {
      log( "*** COMBINE FAILED: " + e.toString() );
      if ( e.stack )
         log( String( e.stack ) );
   }
   logFile.close();
}

function measureMoon( g, W, H, log, fixedR )
{
   let seed = seedDisc( g, W, H, log );
   let grad = logGradient( g, W, H );

   let best;
   if ( fixedR > 0 )
   {
      best = searchCentre( grad, W, H, fixedR, log );
      best.r = fixedR;
      log( "  radius imposed at " + fixedR.toFixed( 1 ) + " px" );
   }
   else
   {
      // Scan radii around the seed, taking the centre search's best score at
      // each. The limb ring outscores everything else, so the peak over radius
      // is the true disc.
      best = { score: -Infinity };
      let rLo = seed.r*R_LO_FRAC, rHi = seed.r*R_HI_FRAC;
      for ( let k = 0; k <= R_SEARCH_STEPS; ++k )
      {
         let R = rLo + (rHi - rLo)*k/R_SEARCH_STEPS;
         let c = searchCentre( grad, W, H, R, null );
         if ( c.score > best.score )
            best = { cx: c.cx, cy: c.cy, r: R, score: c.score };
      }
      log( "  radius scan " + rLo.toFixed( 0 ) + ".." + rHi.toFixed( 0 )
           + " -> " + best.r.toFixed( 1 ) + " px" );
      // Polish the centre at the winning radius on the finest grid.
      let c = searchCentre( grad, W, H, best.r, log );
      best.cx = c.cx; best.cy = c.cy; best.score = c.score;
   }

   log( "  moon centre (" + best.cx.toFixed( 1 ) + ", " + best.cy.toFixed( 1 )
        + ") limb radius " + best.r.toFixed( 1 ) + " px, ring score "
        + best.score.toExponential( 3 ) );
   return best;
}

/*
 * Gradient magnitude of log brightness.
 *
 * Log first: the corona spans orders of magnitude, so a plain gradient is
 * dominated by the bright inner region and the limb's signature would depend on
 * exposure. In log space the limb is a step of roughly constant height at every
 * exposure level, which is what makes one scoring function work across the ladder.
 */
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

/* Mean gradient along the circle of radius R centred at (cx, cy). */
function ringScore( grad, W, H, cx, cy, R, cosT, sinT )
{
   let s = 0, n = 0;
   for ( let k = 0; k < RING_SAMPLES; ++k )
   {
      let x = Math.round( cx + cosT[k]*R );
      let y = Math.round( cy + sinT[k]*R );
      if ( x < 1 || y < 1 || x >= W - 1 || y >= H - 1 )
         continue;
      s += grad[y*W + x];
      ++n;
   }
   // Require most of the ring to be inside the frame, or a circle hanging off
   // the edge can win on a handful of bright samples.
   if ( n < RING_SAMPLES*0.6 )
      return -Infinity;
   return s/n;
}

/*
 * Hierarchical search for the centre at a known radius: a coarse sweep of the
 * whole frame, then successively finer local refinements, then a sub-pixel
 * parabola through the best cell and its neighbours.
 */
function searchCentre( grad, W, H, R, log )
{
   let cosT = new Float64Array( RING_SAMPLES ), sinT = new Float64Array( RING_SAMPLES );
   for ( let k = 0; k < RING_SAMPLES; ++k )
   {
      let a = 2*Math.PI*k/RING_SAMPLES;
      cosT[k] = Math.cos( a ); sinT[k] = Math.sin( a );
   }

   let bx = W/2, by = H/2, bs = -Infinity;
   for ( let cy = 0; cy < H; cy += COARSE_STEP )
      for ( let cx = 0; cx < W; cx += COARSE_STEP )
      {
         let s = ringScore( grad, W, H, cx, cy, R, cosT, sinT );
         if ( s > bs ) { bs = s; bx = cx; by = cy; }
      }

   for ( let i = 0; i < REFINE_STEPS.length; ++i )
   {
      let step = REFINE_STEPS[i], span = (i == 0 ? COARSE_STEP : REFINE_STEPS[i - 1]);
      for ( let dy = -span; dy <= span; dy += step )
         for ( let dx = -span; dx <= span; dx += step )
         {
            let s = ringScore( grad, W, H, bx + dx, by + dy, R, cosT, sinT );
            if ( s > bs ) { bs = s; bx = bx + dx; by = by + dy; }
         }
   }

   // Sub-pixel: parabola through the score at +/-1 px on each axis.
   let sc = bs;
   let sxm = ringScore( grad, W, H, bx - 1, by, R, cosT, sinT );
   let sxp = ringScore( grad, W, H, bx + 1, by, R, cosT, sinT );
   let sym = ringScore( grad, W, H, bx, by - 1, R, cosT, sinT );
   let syp = ringScore( grad, W, H, bx, by + 1, R, cosT, sinT );
   let ddx = (sxm + sxp - 2*sc), ddy = (sym + syp - 2*sc);
   let ox = (ddx < 0) ? 0.5*(sxm - sxp)/ddx : 0;
   let oy = (ddy < 0) ? 0.5*(sym - syp)/ddy : 0;
   if ( !(Math.abs( ox ) <= 1) ) ox = 0;
   if ( !(Math.abs( oy ) <= 1) ) oy = 0;

   if ( log )
      log( "  ring search r=" + R.toFixed( 1 ) + " -> (" + (bx + ox).toFixed( 2 )
           + ", " + (by + oy).toFixed( 2 ) + ") score " + bs.toExponential( 3 ) );
   return { cx: bx + ox, cy: by + oy, score: bs };
}

/*
 * Seed by matched filter: find the dark disc sitting inside a bright ring,
 * searching position and radius together on a heavily decimated image. This only
 * has to bracket the radius for the search above; the centre it returns is not
 * used directly.
 */
function seedDisc( g, W, H, log )
{
   const DS = 32;
   let w = Math.floor( W/DS ), h = Math.floor( H/DS );
   let small = new Float32Array( w*h );
   for ( let y = 0; y < h; ++y )
      for ( let x = 0; x < w; ++x )
      {
         let s = 0;
         for ( let j = 0; j < DS; ++j )
         {
            let row = (y*DS + j)*W + x*DS;
            for ( let i = 0; i < DS; ++i )
               s += g[row + i];
         }
         small[y*w + x] = s/(DS*DS);
      }

   const RING_ANGLES = 32;
   let cosA = new Float64Array( RING_ANGLES ), sinA = new Float64Array( RING_ANGLES );
   for ( let k = 0; k < RING_ANGLES; ++k )
   {
      let a = 2*Math.PI*k/RING_ANGLES;
      cosA[k] = Math.cos( a ); sinA[k] = Math.sin( a );
   }

   let rMin = 4, rMax = Math.floor( Math.min( w, h )/2 ) - 2;
   let best = { cx: W/2, cy: H/2, r: Math.min( W, H )/4, score: -Infinity };
   for ( let r = rMin; r <= rMax; ++r )
   {
      let rIn = r*0.70, rOut = r*1.15;
      for ( let cy = r + 1; cy < h - r - 1; ++cy )
         for ( let cx = r + 1; cx < w - r - 1; ++cx )
         {
            let ring = 0, disc = 0, bad = false;
            for ( let k = 0; k < RING_ANGLES; ++k )
            {
               let xo = Math.round( cx + cosA[k]*rOut ), yo = Math.round( cy + sinA[k]*rOut );
               let xi = Math.round( cx + cosA[k]*rIn ), yi = Math.round( cy + sinA[k]*rIn );
               if ( xo < 0 || yo < 0 || xo >= w || yo >= h ) { bad = true; break; }
               ring += small[yo*w + xo];
               disc += small[yi*w + xi];
            }
            if ( bad )
               continue;
            let score = (ring - disc)/RING_ANGLES;
            if ( score > best.score )
               best = { cx: (cx + 0.5)*DS, cy: (cy + 0.5)*DS, r: r*DS, score: score };
         }
   }
   log( "  seed disc centre (" + best.cx.toFixed( 0 ) + ", " + best.cy.toFixed( 0 )
        + ") r~" + best.r.toFixed( 0 ) + " px" );
   return best;
}

main();
