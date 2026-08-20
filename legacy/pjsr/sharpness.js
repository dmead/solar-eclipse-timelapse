#engine v8

/*
 * sharpness.js - compare stacked results on the one thing that matters here.
 *
 * A whole-image gradient metric is not usable for this question. Stacking lowers
 * noise, and noise carries gradient, so a stack can be genuinely sharper and
 * still score LOWER than a single frame. That would have answered the alignment
 * question backwards.
 *
 * So sharpness is measured only where there is real structure - the brightest
 * pixels and their surroundings, which on a prominence frame is the chromospheric
 * rim - and it is reported alongside the noise measured in blank sky. The
 * meaningful figure is the ratio: edge contrast per unit noise.
 *
 *   -r="...sharpness.js,<a.xisf>,<b.xisf>,...,<logPath>"
 */

function main()
{
   let n = jsArguments.length - 1;
   let logPath = jsArguments[n];

   let logFile = new File;
   logFile.createForWriting( logPath );
   function log( s ) { console.writeln( String( s ) ); logFile.outTextLn( String( s ) ); logFile.flush(); }

   try
   {
      for ( let k = 0; k < n; ++k )
      {
         let p = jsArguments[k];
         let wins = ImageWindow.open( p );
         let win = wins[0];
         for ( let i = 1; i < wins.length; ++i )
            wins[i].forceClose();
         let img = win.mainView.image;
         let w = img.width, h = img.height;

         let a = new Float32Array( w*h );
         img.getSamples( a, new Rect( 0, 0, w, h ), 0 );
         win.forceClose();

         // Structure mask: the top 1% of pixels. On these frames that is the
         // lit rim, which is exactly the detail the drizzle is meant to recover.
         let srt = new Float32Array( a );
         srt.sort();
         let hi = srt[Math.floor( srt.length*0.99 )];
         let sky = srt[Math.floor( srt.length*0.20 )];

         // Sobel-free: sum of squared first differences, restricted to the mask
         // and its immediate neighbourhood.
         let edge = 0, cnt = 0;
         for ( let y = 1; y < h-1; ++y )
            for ( let x = 1; x < w-1; ++x )
            {
               let i = y*w + x;
               if ( a[i] < hi )
                  continue;
               let gx = a[i+1] - a[i-1];
               let gy = a[i+w] - a[i-w];
               edge += gx*gx + gy*gy;
               ++cnt;
            }
         edge = cnt ? Math.sqrt( edge/cnt ) : 0;

         // Noise from blank sky: the median absolute difference between
         // horizontally adjacent pixels in the darkest fifth of the frame.
         let devs = [];
         for ( let y = 1; y < h-1; y += 3 )
            for ( let x = 1; x < w-1; x += 3 )
            {
               let i = y*w + x;
               if ( a[i] > sky )
                  continue;
               devs.push( Math.abs( a[i+1] - a[i] ) );
            }
         devs.sort( function( u, v ) { return u - v; } );
         let noise = devs.length ? devs[devs.length >> 1]/0.6745/Math.SQRT2 : 0;

         let name = p.substring( p.lastIndexOf( "/" ) + 1 );
         log( name + "  " + w + "x" + h
            + "  edge=" + edge.toExponential( 4 )
            + "  noise=" + noise.toExponential( 4 )
            + "  edge/noise=" + ( noise > 0 ? (edge/noise).toFixed( 2 ) : "inf" )
            + "  maskPx=" + cnt );
      }
      log( "=== SHARPNESS OK ===" );
   }
   catch ( e )
   {
      log( "ERROR: " + e );
   }
   logFile.close();
}

main();
