#engine v8

/*
 * crop-export.js - export a crop around the brightest structure, unscaled.
 *
 * For eyeballing a stacking or alignment trial. The crop is centred on the
 * centroid of the brightest 0.5% of pixels, which on a prominence frame lands on
 * the chromospheric rim, and every file in one run is normalised by the SAME
 * scale factor - taken from the first file - so brightness differences between
 * them are real and not an artefact of per-image autoscaling.
 *
 *   -r="...crop-export.js,<size>,<gamma>,<a.xisf>,<b.xisf>,...,<outDir>,<logPath>"
 */

function main()
{
   let n = jsArguments.length;
   let logPath = jsArguments[n-1];
   let outDir = jsArguments[n-2];
   let size = parseInt( jsArguments[0], 10 );
   let gamma = parseFloat( jsArguments[1] );

   let logFile = new File;
   logFile.createForWriting( logPath );
   function log( s ) { console.writeln( String( s ) ); logFile.outTextLn( String( s ) ); logFile.flush(); }

   try
   {
      let cx = -1, cy = -1, norm = 0;

      for ( let k = 2; k < n-2; ++k )
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

         if ( cx < 0 )
         {
            // Locate once, on the first file, so every crop shows the same patch.
            let srt = new Float32Array( a );
            srt.sort();
            let hi = srt[Math.floor( srt.length*0.995 )];
            norm = srt[srt.length-1];
            let sx = 0, sy = 0, sw = 0;
            for ( let y = 0; y < h; ++y )
               for ( let x = 0; x < w; ++x )
               {
                  let v = a[y*w + x];
                  if ( v < hi )
                     continue;
                  sx += x*v; sy += y*v; sw += v;
               }
            cx = Math.round( sx/sw ); cy = Math.round( sy/sw );
            log( "crop centred on (" + cx + ", " + cy + "), " + size + "x" + size );
         }

         let x0 = Math.max( 0, Math.min( w - size, cx - (size >> 1) ) );
         let y0 = Math.max( 0, Math.min( h - size, cy - (size >> 1) ) );
         let out = new Float32Array( size*size );
         for ( let y = 0; y < size; ++y )
            for ( let x = 0; x < size; ++x )
            {
               let v = a[(y0 + y)*w + x0 + x]/norm;
               out[y*size + x] = Math.pow( Math.max( 0, Math.min( 1, v ) ), gamma );
            }

         let cw = new ImageWindow( size, size, 1, 8, false, false, "crop" );
         // UndoFlag_NoSwapFile is a <pjsr/UndoFlag.jsh> constant, and that header
         // does not load under #engine v8 here. -1 is the same "no swap" value.
         cw.mainView.beginProcess( -1 );
         cw.mainView.image.setSamples( out );
         cw.mainView.endProcess();
         let name = p.substring( p.lastIndexOf( "/" ) + 1 ).replace( ".xisf", "" );
         cw.saveAs( outDir + "/" + name + "_crop.png", false, false, false, false );
         cw.forceClose();
         log( "  wrote " + name + "_crop.png" );
      }
      log( "=== CROP OK ===" );
   }
   catch ( e )
   {
      log( "ERROR: " + e );
   }
   logFile.close();
}

main();
