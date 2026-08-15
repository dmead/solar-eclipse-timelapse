/*
 * ser-slice.js — extract a contiguous frame range from a SER capture.
 *
 * An eclipse SER holds several exposure states in one file: the filter comes off
 * mid-capture and the exposure is then ridden down by hand. ser-stack.js consumes
 * a whole file, so the segments found by Stage A have to be split out first.
 *
 * Unlike pix-planetary's ser-trim.js this does NOT quality-rank or ROI-crop —
 * both of its heuristics assume a lit lunar disk and are wrong for a corona. It is
 * a straight byte-range copy that preserves colorId, bit depth, geometry (so CFA
 * phase is trivially intact) and the matching slice of trailer timestamps, which
 * ser-trim.js discards.
 *
 * Frames are streamed in bounded chunks — these files are up to 22 GB.
 *
 *   -r="...ser-slice.js,<in.ser>,<out.ser>,<start>,<count>,<logPath>"
 */

#engine v8

const DT_INT32 = 5;
const DT_BYTEARRAY = 15;

const HEADER_BYTES = 178;
const CHUNK_TARGET = 64 * 1024 * 1024; // cap RAM per read regardless of frame size

function main()
{
   let inPath = jsArguments[0];
   let outPath = jsArguments[1];
   let start = parseInt( jsArguments[2], 10 );
   let count = parseInt( jsArguments[3], 10 );
   let logPath = jsArguments[4];

   // Create the log first: a missing PJSR API kills the script silently with
   // exit 0, so "no log file" has to mean "died on the first line", not "never ran".
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
      let header = f.read( DT_BYTEARRAY, HEADER_BYTES );
      f.position = 18;
      let a = f.read( DT_INT32, 6 );
      let colorId = a[0], W = a[2], H = a[3], depth = a[4], frameCount = a[5];
      let planes = colorId >= 100 ? 3 : 1;
      let bpp = depth > 8 ? 2 : 1;
      let frameBytes = W * H * planes * bpp;
      let inSize = f.size;

      let usable = Math.min( frameCount, Math.floor( (inSize - HEADER_BYTES)/frameBytes ) );
      let trailerOff = HEADER_BYTES + frameCount*frameBytes;
      let hasTrailer = inSize >= trailerOff + frameCount*8;

      log( "slice " + inPath );
      log( "  source " + W + "x" + H + " " + depth + "b colorId=" + colorId
           + " frames=" + frameCount + " usable=" + usable
           + " trailer=" + hasTrailer );

      if ( !(start >= 0) || !(count > 0) )
         throw new Error( "bad range: start=" + start + " count=" + count );
      if ( start >= usable )
         throw new Error( "start " + start + " beyond usable frame count " + usable );
      if ( start + count > usable )
      {
         log( "  WARNING: range truncated to usable frames ("
              + count + " -> " + (usable - start) + ")" );
         count = usable - start;
      }

      let out = new File;
      out.createForWriting( outPath );
      out.write( header );
      out.position = 38;
      out.write( count, DT_INT32 );   // rewrite frameCount for the slice
      out.position = HEADER_BYTES;

      // ---- frame data, streamed ----
      let chunkFrames = Math.max( 1, Math.floor( CHUNK_TARGET/frameBytes ) );
      let done = 0;
      while ( done < count )
      {
         let k = Math.min( chunkFrames, count - done );
         f.position = HEADER_BYTES + (start + done)*frameBytes;
         out.write( f.read( DT_BYTEARRAY, k*frameBytes ) );
         done += k;
         if ( done % (chunkFrames*8) < chunkFrames )
            log( "  copied " + done + "/" + count + " frames" );
      }

      // ---- trailer: the matching timestamp slice ----
      // ser-trim.js drops these; downstream stages (and the Python reader) treat
      // the trailer as authoritative for frame times, so they are carried over.
      if ( hasTrailer )
      {
         f.position = trailerOff + start*8;
         out.write( f.read( DT_BYTEARRAY, count*8 ) );
      }
      else
         log( "  note: source had no trailer — slice has none either" );

      out.close();
      f.close();

      // The header's DateTime fields still describe the original capture start.
      // That is deliberate: rewriting them needs int64 arithmetic PJSR makes
      // awkward, and every consumer here reads frame times from the trailer.
      let outSize = HEADER_BYTES + count*frameBytes + (hasTrailer ? count*8 : 0);
      log( "  wrote " + count + " frames [" + start + ".." + (start + count - 1) + "] "
           + (outSize/1e9).toFixed( 2 ) + " GB in "
           + ((Date.now() - t0)/1000).toFixed( 0 ) + " s" );
      log( "=== SLICE OK ===" );
   }
   catch ( e )
   {
      log( "*** SLICE FAILED: " + e.toString() );
      if ( e.stack )
         log( String( e.stack ) );
   }
   logFile.close();
}

main();
