/*
 * ser-frames.js — export individual SER frames as colour images.
 *
 * Baily's beads and the diamond ring evolve in well under a second, so unlike
 * every other product here they must NOT be stacked — stacking averages the beads
 * into a smooth arc. This pulls single frames out instead.
 *
 * Demosaic is bilinear at full resolution rather than 2x2 superpixel: beads are
 * small high-contrast features and halving the resolution to avoid interpolation
 * costs more than the interpolation does.
 *
 *   -r="...ser-frames.js,<in.ser>,<start>,<count>,<stride>,<outDir>,<prefix>,<mode>,<logPath>"
 *
 * mode "preview" writes downsampled autostretched 8-bit PNGs for triage;
 * mode "tif" writes full-resolution 16-bit TIFFs of the frames worth keeping.
 */

#engine v8

const DT_INT32 = 5;
const DT_UINT16ARRAY = 23;
const DT_UINT8ARRAY = 25;

const HEADER_BYTES = 178;
const PREVIEW_WIDTH = 960;

// CFA site offsets [x,y] within each 2x2 cell, per SER colorId.
const CFA_LAYOUT = {
   8:  { R: [0, 0], G1: [1, 0], G2: [0, 1], B: [1, 1] },   // RGGB
   9:  { R: [1, 0], G1: [0, 0], G2: [1, 1], B: [0, 1] },   // GRBG
   10: { R: [0, 1], G1: [0, 0], G2: [1, 1], B: [1, 0] },   // GBRG
   11: { R: [1, 1], G1: [1, 0], G2: [0, 1], B: [0, 0] },   // BGGR
};

function main()
{
   let inPath = jsArguments[0];
   let start = parseInt( jsArguments[1], 10 );
   let count = parseInt( jsArguments[2], 10 );
   let stride = Math.max( 1, parseInt( jsArguments[3], 10 ) );
   let outDir = jsArguments[4];
   let prefix = jsArguments[5];
   let mode = jsArguments[6];
   let logPath = jsArguments[7];

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
      let f = new File;
      f.openForReading( inPath );
      f.position = 18;
      let a = f.read( DT_INT32, 6 );
      let colorId = a[0], W = a[2], H = a[3], depth = a[4], frameCount = a[5];
      let bpp = depth > 8 ? 2 : 1;
      let maxv = depth == 16 ? 65535 : 255;
      let frameBytes = W*H*bpp;
      let cfa = CFA_LAYOUT[colorId];
      if ( !cfa )
         throw new Error( "colorId " + colorId + " is not a supported Bayer layout" );

      if ( start + count > frameCount )
         count = frameCount - start;
      log( "frames " + inPath );
      log( "  " + W + "x" + H + " colorId=" + colorId + " -> " + mode
           + ", frames [" + start + ".." + (start + count - 1) + "] stride " + stride );

      if ( !File.directoryExists( outDir ) )
         File.createDirectory( outDir, true );

      let n = W*H;
      let R = new Float32Array( n ), G = new Float32Array( n ), B = new Float32Array( n );
      let written = 0;

      for ( let k = 0; k < count; k += stride )
      {
         let fi = start + k;
         f.position = HEADER_BYTES + fi*frameBytes;
         let s = f.read( bpp == 2 ? DT_UINT16ARRAY : DT_UINT8ARRAY, n );
         demosaic( s, W, H, maxv, cfa, R, G, B );

         // The identifier is internal only, and PixInsight requires a valid JS
         // identifier: these captures are named by clock time, so anything built
         // from the prefix starts with a digit and is rejected.
         let win = new ImageWindow( W, H, 3, mode == "tif" ? 16 : 8, false, true,
                                    "ecframe" );
         let v = win.mainView;
         v.beginProcess( UndoFlag.NoSwapFile );
         v.image.selectedChannel = 0; v.image.setSamples( R );
         v.image.selectedChannel = 1; v.image.setSamples( G );
         v.image.selectedChannel = 2; v.image.setSamples( B );
         v.image.resetSelections();

         let name;
         if ( mode == "preview" )
         {
            // Autostretch for triage only — these are for choosing frames by eye,
            // never a deliverable.
            v.image.rescale();
            v.image.apply( 0.45, ImageOp.Pow );
            v.image.interpolation = InterpolationAlgorithm.MitchellNetravaliFilter;
            v.image.resample( PREVIEW_WIDTH/W );
            name = outDir + "/" + prefix + "_" + pad( fi ) + ".png";
         }
         else
            name = outDir + "/" + prefix + "_" + pad( fi ) + ".tif";
         v.endProcess();

         if ( !win.saveAs( name, false, false, false, false ) )
            throw new Error( "save failed: " + name );
         win.forceClose();
         if ( ++written % 20 == 0 )
            log( "  wrote " + written + " frames" );
      }

      f.close();
      log( "  " + written + " frames -> " + outDir + " in "
           + ((Date.now() - t0)/1000).toFixed( 0 ) + " s" );
      log( "=== FRAMES OK ===" );
   }
   catch ( e )
   {
      log( "*** FRAMES FAILED: " + e.toString() );
      if ( e.stack )
         log( String( e.stack ) );
   }
   logFile.close();
}

function pad( i )
{
   let s = String( i );
   while ( s.length < 5 )
      s = "0" + s;
   return s;
}

/*
 * Bilinear CFA interpolation, normalized to [0,1].
 * Each output pixel takes its own colour directly when it sits on that colour's
 * site, and the mean of the nearest same-colour sites otherwise. Coordinates are
 * clamped at the borders (mirroring would fold the CFA phase and swap colours).
 */
function demosaic( s, W, H, maxv, cfa, R, G, B )
{
   let inv = 1/maxv;
   let rx = cfa.R[0], ry = cfa.R[1];
   let bx = cfa.B[0], by = cfa.B[1];

   function at( x, y )
   {
      if ( x < 0 ) x = 0; else if ( x >= W ) x = W - 1;
      if ( y < 0 ) y = 0; else if ( y >= H ) y = H - 1;
      return s[y*W + x];
   }

   for ( let y = 0; y < H; ++y )
   {
      let row = y*W;
      let py = y & 1;
      for ( let x = 0; x < W; ++x )
      {
         let px = x & 1, i = row + x;
         let isR = (px == rx && py == ry);
         let isB = (px == bx && py == by);

         if ( isR )
         {
            R[i] = s[i]*inv;
            G[i] = (at( x - 1, y ) + at( x + 1, y ) + at( x, y - 1 ) + at( x, y + 1 ))*0.25*inv;
            B[i] = (at( x - 1, y - 1 ) + at( x + 1, y - 1 )
                  + at( x - 1, y + 1 ) + at( x + 1, y + 1 ))*0.25*inv;
         }
         else if ( isB )
         {
            B[i] = s[i]*inv;
            G[i] = (at( x - 1, y ) + at( x + 1, y ) + at( x, y - 1 ) + at( x, y + 1 ))*0.25*inv;
            R[i] = (at( x - 1, y - 1 ) + at( x + 1, y - 1 )
                  + at( x - 1, y + 1 ) + at( x + 1, y + 1 ))*0.25*inv;
         }
         else
         {
            // Green site: the other two colours lie along one axis each, and
            // which axis holds red depends on whether this row carries red.
            G[i] = s[i]*inv;
            if ( py == ry )
            {
               R[i] = (at( x - 1, y ) + at( x + 1, y ))*0.5*inv;
               B[i] = (at( x, y - 1 ) + at( x, y + 1 ))*0.5*inv;
            }
            else
            {
               R[i] = (at( x, y - 1 ) + at( x, y + 1 ))*0.5*inv;
               B[i] = (at( x - 1, y ) + at( x + 1, y ))*0.5*inv;
            }
         }
      }
   }
}

main();
