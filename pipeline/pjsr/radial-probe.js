#engine v8   // keep within the first few lines: PixInsight only scans a short
             // prologue for this, and a longer header comment pushes it out of
             // range, leaving the script in ES5 where let and arrow functions
             // are a load error that -r mode discards silently with exit 0.

/*
 * radial-probe.js - dump an azimuthally averaged radial profile as CSV.
 *
 * Diagnostic only. A ring or step in a processed corona image is a change in how
 * the pixel was made, not something the Sun did, and the radius at which it
 * happens identifies which stage did it: a merge weight ramping out, a feather
 * boundary, or the edge of where a profile was measurable.
 *
 *   -r="...radial-probe.js,<in.xisf>,<cx>,<cy>,<out.csv>,<logPath>"
 */

function main()
{
   let inPath = jsArguments[0];
   let cx = parseFloat( jsArguments[1] );
   let cy = parseFloat( jsArguments[2] );
   let outPath = jsArguments[3];
   let logPath = jsArguments[4];

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
      let wins = ImageWindow.open( inPath );
      let win = wins[0];
      for ( let i = 1; i < wins.length; ++i )
         wins[i].forceClose();
      let img = win.mainView.image;
      let W = img.width, H = img.height;
      let g = new Float32Array( W*H );
      img.getSamples( g, new Rect( 0, 0, W, H ), img.numberOfChannels > 1 ? 1 : 0 );
      win.forceClose();
      log( "probe " + inPath + " " + W + "x" + H + " about ("
           + cx.toFixed( 1 ) + ", " + cy.toFixed( 1 ) + ")" );

      let rmax = Math.ceil( Math.sqrt( Math.max( cx, W - cx )*Math.max( cx, W - cx )
                                     + Math.max( cy, H - cy )*Math.max( cy, H - cy ) ) );
      let sum = new Float64Array( rmax + 1 );
      let cnt = new Int32Array( rmax + 1 );
      for ( let y = 0; y < H; ++y )
      {
         let dy = y - cy, row = y*W;
         for ( let x = 0; x < W; ++x )
         {
            let dx = x - cx;
            let r = Math.round( Math.sqrt( dx*dx + dy*dy ) );
            if ( r <= rmax ) { sum[r] += g[row + x]; cnt[r]++; }
         }
      }

      let out = new File;
      out.createForWriting( outPath );
      out.outTextLn( "r,mean,count" );
      for ( let r = 0; r <= rmax; ++r )
         out.outTextLn( r + "," + (cnt[r] > 0 ? (sum[r]/cnt[r]).toExponential( 6 ) : "0")
                        + "," + cnt[r] );
      out.close();

      log( "  wrote " + outPath + " (" + rmax + " radii)" );
      log( "=== PROBE OK ===" );
   }
   catch ( e )
   {
      log( "*** PROBE FAILED: " + e.toString() );
      if ( e.stack )
         log( String( e.stack ) );
   }
   logFile.close();
}

main();
