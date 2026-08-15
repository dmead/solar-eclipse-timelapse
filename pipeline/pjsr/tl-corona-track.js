#engine v8

/*
 * tl-corona-track.js - measure the pointing during totality by correlating the
 * corona against itself.
 *
 * The disc fit that stabilises the partial phases has nothing to fit during
 * totality: there is no photosphere. tl-centres.js falls back to scoring
 * candidate centres on the gradient around the Moon's limb, and that scatters
 * about 7 px with a lag-1 autocorrelation of only +0.2 to +0.5 - noise rather
 * than motion - so smooth_track.py refuses to follow it and places totality on a
 * straight line. The line removes the drift. It cannot remove the mount's actual
 * wobble, and what it leaves behind is visible in the video, magnified threefold
 * inside the zoom panels: measured against this pass, the line sits 0.74 px RMS
 * from where the picture says the frame is.
 *
 * Phase correlation does not have that problem. The frame is not a smooth blob:
 * it carries the lunar limb, a hard edge right around the disc, and the
 * prominences on it. That is exactly what the drizzle already relies on to stack
 * twenty raw frames inside one group - this measures the same quantity BETWEEN
 * groups, which is the part nothing was measuring.
 *
 * Every frame of a capture is measured against ONE reference frame rather than
 * against its predecessor. A chain of frame-to-frame shifts accumulates its own
 * errors, and over seventy frames a small bias becomes a drift that looks exactly
 * like the thing being corrected. A common reference cannot accumulate. The mount
 * was nudged between captures, so no reference is carried across one.
 *
 * The reference is the middle frame of the capture's LONGEST constant-exposure
 * run, not simply its first frame. Phase correlation is indifferent to a change
 * of gain, because it throws away magnitude and keeps only phase - but it is not
 * indifferent to CLIPPING, which is a change of structure rather than of scale,
 * and the inner corona clips at the long exposures. 14_14_36 opens on a transition
 * frame exposed at half the level the rest of the capture settles to. (Measured:
 * this made almost no difference here, 0.539 to 0.531 px - the theory was wrong
 * about what ailed that capture, which turned out to be a metric artifact. Kept
 * because it is right in principle and costs nothing.)
 *
 *   -r="...tl-corona-track.js,<timelapse.json>,<out.json>,<logPath>"
 */

#include "D:/projects/pix-planetary/pjsr/lib/fftalign.jsh"

const DT_INT32 = 5;
const DT_UINT16ARRAY = 23;
const DT_UINT8ARRAY = 25;
const HEADER_BYTES = 178;

// A shift larger than this is a failed correlation, not the mount. The mount
// drifts ~40 px over a 60 s capture, so this has to be generous.
const MAX_SHIFT_PX = 80.0;

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

      // One entry per DISTINCT raw frame. The prominence level is resampled with
      // repeats to hold it on screen, and correlating the same frame twice is
      // wasted work - the answer cannot differ.
      let want = [], seen = {};
      for ( let k = 0; k < cfg.frames.length; ++k )
      {
         let fr = cfg.frames[k];
         if ( fr.state != "unfiltered" )
            continue;
         let key = fr.file + "#" + fr.index;
         if ( seen[key] )
            continue;
         seen[key] = true;
         want.push( fr );
      }
      log( "corona track: " + want.length + " distinct totality frames" );

      // Group by capture, in order, and choose each one's reference.
      let order = [], byFile = {};
      for ( let k = 0; k < want.length; ++k )
      {
         let fr = want[k];
         if ( !byFile[fr.file] )
         {
            byFile[fr.file] = [];
            order.push( fr.file );
         }
         byFile[fr.file].push( fr );
      }

      let results = [];
      let failed = 0, done = 0;

      for ( let f = 0; f < order.length; ++f )
      {
         let rows = byFile[order[f]];
         rows.sort( function ( a, b ) { return a.index - b.index; } );

         // Modal exposure level, then the middle frame of it.
         let count = {}, bestKey = null, bestN = 0;
         for ( let k = 0; k < rows.length; ++k )
         {
            let g = String( Math.round( rows[k].gain*1e4 ) );
            count[g] = (count[g] || 0) + 1;
            if ( count[g] > bestN ) { bestN = count[g]; bestKey = g; }
         }
         let run = [];
         for ( let k = 0; k < rows.length; ++k )
            if ( String( Math.round( rows[k].gain*1e4 ) ) == bestKey )
               run.push( rows[k] );
         let ref = run[run.length >> 1];

         let cur = new File;
         cur.openForReading( rows[0].src );
         cur.position = 18;
         let a = cur.read( DT_INT32, 6 );
         let W = a[2], H = a[3];
         let maxv = a[4] == 16 ? 65535 : 255;
         let frameBytes = W*H*(a[4] > 8 ? 2 : 1);
         let w = W >> 1, h = H >> 1;
         let G = new Float32Array( w*h );
         log( "  " + order[f] + ": " + rows.length + " frames, reference f"
            + ref.index + " (level " + ref.gain + ", " + bestN + " frames)" );

         readG( cur, ref.index, W, H, w, h, frameBytes, maxv, G );
         let refIm = toImage( G, w, h );
         let aligner = new FFTTranslation( false );
         aligner.initialize( refIm );
         refIm.free();

         for ( let k = 0; k < rows.length; ++k )
         {
            let fr = rows[k];
            if ( fr.index == ref.index )
            {
               results.push( { file: fr.file, index: fr.index,
                               dx: 0, dy: 0, ok: true } );
            }
            else
            {
               readG( cur, fr.index, W, H, w, h, frameBytes, maxv, G );
               let im = toImage( G, w, h );
               let sh = aligner.evaluate( im );
               im.free();
               let ok = Math.abs( sh.dx ) <= MAX_SHIFT_PX
                     && Math.abs( sh.dy ) <= MAX_SHIFT_PX;
               if ( !ok )
                  ++failed;
               results.push( { file: fr.file, index: fr.index,
                               dx: ok ? sh.dx : 0, dy: ok ? sh.dy : 0, ok: ok } );
            }
            if ( ++done % 100 == 0 )
               log( "  " + done + "/" + want.length );
         }
         aligner.free();
         cur.close();
      }

      let out = new File;
      out.createForWriting( outPath );
      out.outText( JSON.stringify( { shifts: results, maxShift: MAX_SHIFT_PX } ) );
      out.close();

      log( "  " + results.length + " measured, " + failed + " rejected, in "
           + ((Date.now() - t0)/60000).toFixed( 1 ) + " min" );
      log( "=== CORONA TRACK OK ===" );
   }
   catch ( e )
   {
      log( "*** CORONA TRACK FAILED: " + e.toString() );
      if ( e.stack )
         log( String( e.stack ) );
   }
   logFile.close();
}

/* Superpixel G only - the same half-resolution grid the disc track lives on. */
function readG( cur, index, W, H, w, h, frameBytes, maxv, G )
{
   cur.position = HEADER_BYTES + index*frameBytes;
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
}

function toImage( arr, w, h )
{
   let im = Image.newFloatImage();
   im.allocate( w, h );
   im.setSamples( arr );
   return im;
}

main();
