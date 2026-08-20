#engine v8

/*
 * xisf-export.js - write an XISF to PNG with NO tone changes.
 *
 * pix-planetary's xisf-preview.js rescales and applies a 0.6 gamma, which is
 * right for inspecting a linear stack but wrong for judging an image that has
 * already been stretched: it re-stretches on top and the result is not what the
 * file contains. This just downsamples and writes what is there.
 *
 *   -r="...xisf-export.js,<in.xisf>,<out.png>,<maxWidth>,<logPath>"
 */

function main()
{
   let inPath = jsArguments[0];
   let outPath = jsArguments[1];
   let maxW = parseInt( jsArguments[2], 10 );
   let logPath = jsArguments[3];

   let logFile = new File;
   logFile.createForWriting( logPath );
   function log( s ) { logFile.outTextLn( String( s ) ); logFile.flush(); }

   try
   {
      let wins = ImageWindow.open( inPath );
      let win = wins[0];
      for ( let i = 1; i < wins.length; ++i )
         wins[i].forceClose();
      let v = win.mainView;
      let W = v.image.width;
      if ( maxW > 0 && W > maxW )
      {
         v.beginProcess( UndoFlag.NoSwapFile );
         v.image.interpolation = InterpolationAlgorithm.MitchellNetravaliFilter;
         v.image.resample( maxW/W );
         v.endProcess();
      }
      win.setSampleFormat( 8, false );
      if ( !win.saveAs( outPath, false, false, false, false ) )
         throw new Error( "save failed: " + outPath );
      win.forceClose();
      log( "exported " + outPath );
      log( "=== EXPORT OK ===" );
   }
   catch ( e )
   {
      log( "*** EXPORT FAILED: " + e.toString() );
   }
   logFile.close();
}

main();
