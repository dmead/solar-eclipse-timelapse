/*
 * tl-frames.js — render timelapse frames: demosaic, centre the Sun, apply the
 * per-frame gain chosen by gen_timelapse.py, write 8-bit PNGs for ffmpeg.
 *
 * Demosaic is 2x2 superpixel, which lands exactly on 1920x1080 from this sensor
 * and costs no interpolation — the right trade for video, where the extra
 * resolution would be thrown away by the encoder anyway.
 *
 * Centring matters more than it looks: the mount drifts over the 45 minutes of
 * capture, and an uncentred Sun makes the video wander. Each frame is recentred
 * on its own measured centroid.
 *
 *   -r="...tl-frames.js,<config.json>,<logPath>"
 */

#engine v8

#include "D:/projects/pix-planetary/pjsr/lib/fftalign.jsh"

const DT_INT32 = 5;
const DT_UINT16ARRAY = 23;
const DT_UINT8ARRAY = 25;
const HEADER_BYTES = 178;

/*
 * Every frame is rendered on a 2x grid - the sensor's own sampling, not the
 * superpixel grid the disc fits are measured on.
 *
 * For a frame carrying a "stack" count that grid is filled by drizzle: N
 * consecutive raw frames, phase-correlated against the first and added onto the
 * fine grid, so the sub-pixel dither the mount supplied recovers detail past the
 * native sampling. That matters for the prominences specifically. They are
 * Halpha, so they live almost entirely in R, and R is only a quarter of the CFA
 * sites - sampled at 3.43 arcsec/px, half the linear resolution of luminance and
 * the worst-sampled signal in the capture. Drizzle is the only thing that
 * recovers it; a better demosaic cannot, because R is not interpolated away, it
 * is genuinely that sparse.
 *
 * Frames with no stack count are simply resampled onto the same grid. They gain
 * nothing, but the video needs one geometry throughout, and rendering everything
 * at 2x is what lets the drizzled totality sit in the same sequence as the
 * partials.
 */
const DRIZZLE = 2;
const INTERP = (typeof InterpolationAlgorithm != "undefined")
             ? InterpolationAlgorithm.BicubicSpline : 2;
// A group spans 0.86 s, over which the measured pointing moves under 1 px. A
// shift far beyond that is a failed correlation, not motion, and the frame is
// left out rather than smeared into the stack.
const MAX_GROUP_SHIFT_PX = 4.0;

// Display gamma applied after the linear gain. The corona and the partial disc
// have very different structure; a mild lift keeps faint corona visible without
// making the filtered partials look washed out.
const GAMMA = 0.65;

// Fraction of brightest pixels whose centroid defines the Sun's position.
const CENTROID_FRACTION = 0.02;

/*
 * Highlight shoulder.
 *
 * The gain maps each segment's p99 to a fixed target, but p99 is not the peak:
 * on a nearly full disc the photosphere covers enough of the frame that p99 sits
 * INSIDE it, and the brighter centre lands above 1.0 and hard-clips. Measured in
 * the render, 13% of the first frame was pinned at 254+ while the source frames
 * themselves clip nothing - the flat white disc was manufactured here.
 *
 * Below the knee the response stays linear, so nothing that was correctly exposed
 * is altered. Above it, tanh compresses asymptotically toward 1.0 and never quite
 * reaches it, so limb darkening and sunspots near disc centre keep their gradient
 * instead of being flattened into a white plate.
 */
const SHOULDER_KNEE = 0.60;

/*
 * Ceiling the shoulder approaches. Deliberately below 1.0: a shoulder that
 * asymptotes TO white still renders 255 for anything far enough over the knee,
 * which is what left 9% of the fullest disc pinned after the first attempt. With
 * the ceiling at 0.965 the brightest possible pixel lands at 249 after the 0.65
 * display gamma, so the disc always keeps some headroom and the eye can still see
 * its gradient.
 */
const SHOULDER_CEIL = 0.965;

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
      let frames = cfg.frames;
      let outDir = cfg.outDir;
      if ( !File.directoryExists( outDir ) )
         File.createDirectory( outDir, true );

      log( "timelapse: " + frames.length + " frames -> " + outDir );

      // Output window, cropped from the demosaiced frame around the Sun.
      let outW = cfg.outW || 1280, outH = cfg.outH || 720;
      let outW2 = DRIZZLE*outW, outH2 = DRIZZLE*outH;
      let DISSOLVE = (cfg.dissolve !== undefined) ? cfg.dissolve : 3;
      let dissolveFile = null, dissolveLeft = 0, dissolveSrc = null, prevOut = null;
      let dissolveIndex = -1;
      // Frame gap inside one capture that counts as a discontinuity.
      const GAP_FRAMES = 60;
      log( "  output " + outW2 + "x" + outH2 + " (drizzle x" + DRIZZLE
         + "), dissolve " + DISSOLVE + " frames" );

      let cur = null, curPath = "";
      let W = 0, H = 0, w = 0, h = 0, frameBytes = 0, maxv = 65535;
      let R = null, G = null, B = null;

      for ( let k = 0; k < frames.length; ++k )
      {
         let fr = frames[k];
         if ( fr.src != curPath )
         {
            if ( cur ) cur.close();
            cur = new File;
            cur.openForReading( fr.src );
            cur.position = 18;
            let a = cur.read( DT_INT32, 6 );
            W = a[2]; H = a[3];
            maxv = a[4] == 16 ? 65535 : 255;
            frameBytes = W*H*(a[4] > 8 ? 2 : 1);
            w = W >> 1; h = H >> 1;
            R = new Float32Array( w*h ); G = new Float32Array( w*h ); B = new Float32Array( w*h );
            curPath = fr.src;
            log( "  open " + fr.file + " (" + w + "x" + h + ")" );
         }

         readSuperpixel( cur, fr.index, W, H, w, h, frameBytes, maxv, R, G, B );

         /*
          * Drizzle the group onto the fine grid.
          *
          * Alignment is measured once, on G, and applied to all three channels.
          * Per-channel alignment would let the colours drift apart by a fraction
          * of a pixel, and G carries the most signal so it gives the best
          * correlation of the three anyway.
          *
          * The reference is the group's FIRST frame, which is also the frame the
          * disc track was measured on - so the stacked result sits exactly where
          * the unstacked one would have, and stacked and unstacked frames can be
          * mixed in one sequence without the subject stepping.
          */
         let fine = [ null, null, null ];
         let nStack = 1;
         if ( fr.stack > 1 )
         {
            let refIm = toImage( G, w, h );
            let aligner = new FFTTranslation( false );
            aligner.initialize( refIm );
            refIm.free();

            let acc = [ accumulate( R, w, h, 0, 0, null ),
                        accumulate( G, w, h, 0, 0, null ),
                        accumulate( B, w, h, 0, 0, null ) ];
            let R2 = new Float32Array( w*h ), G2 = new Float32Array( w*h ),
                B2 = new Float32Array( w*h );
            for ( let j = 1; j < fr.stack; ++j )
            {
               readSuperpixel( cur, fr.index + j, W, H, w, h, frameBytes, maxv,
                               R2, G2, B2 );
               let gIm = toImage( G2, w, h );
               let sh = aligner.evaluate( gIm );
               gIm.free();
               if ( Math.abs( sh.dx ) > MAX_GROUP_SHIFT_PX
                 || Math.abs( sh.dy ) > MAX_GROUP_SHIFT_PX )
                  continue;
               let src = [ R2, G2, B2 ];
               for ( let c = 0; c < 3; ++c )
                  accumulate( src[c], w, h, -sh.dx, -sh.dy, acc[c] );
               ++nStack;
            }
            aligner.free();
            for ( let c = 0; c < 3; ++c )
            {
               acc[c].apply( nStack, ImageOp.Div );
               fine[c] = new Float32Array( DRIZZLE*w*DRIZZLE*h );
               acc[c].getSamples( fine[c] );
               acc[c].free();
            }
         }
         else
         {
            let src = [ R, G, B ];
            for ( let c = 0; c < 3; ++c )
            {
               let im = accumulate( src[c], w, h, 0, 0, null );
               fine[c] = new Float32Array( DRIZZLE*w*DRIZZLE*h );
               im.getSamples( fine[c] );
               im.free();
            }
         }

         /*
          * Hold the Sun still.
          *
          * The centre comes from the smoothed disc track, not from this frame:
          * per-frame detection jitters by a pixel or two and gives up entirely on
          * the thinnest crescents, and either shows up as the picture twitching.
          *
          * The window is NOT clamped inside the sensor. Clamping would keep the
          * frame full at the cost of letting the Sun slide - which is the whole
          * problem - and it also crops away corona that the sensor did record.
          * Where the window reaches past the edge the renderer pads black, which
          * is the honest statement that nothing was captured there.
          */
         let ox, oy;
         if ( fr.cx !== undefined )
         {
            ox = fr.cx - outW/2;
            oy = fr.cy - outH/2;
         }
         else
         {
            let ctr = centroid( G, w, h );
            ox = clamp( ctr.cx - outW/2, 0, w - outW );
            oy = clamp( ctr.cy - outH/2, 0, h - outH );
         }

         // Crop on the fine grid. The disc track is measured in superpixels, so
         // both the centre and the window scale by the drizzle factor.
         let outPlanes = [];
         for ( let c = 0; c < 3; ++c )
            outPlanes.push( cropGain( fine[c], DRIZZLE*w, DRIZZLE*h,
                                      DRIZZLE*ox, DRIZZLE*oy,
                                      outW2, outH2, fr.gain ) );

         /*
          * Cross-dissolve across a capture boundary.
          *
          * Within a capture successive video frames are 0.86 s of real time
          * apart and the Moon advances 0.24 px. At a boundary the recorder was
          * flushing to disk for 25 to 490 s, so the Moon jumps 7 to 136 px in a
          * single frame - thirty to five hundred times the normal step, and it
          * reads as a jolt. Nothing can fill that gap; there is no data. A short
          * dissolve just stops the discontinuity landing on one frame.
          */
         // Any real-time gap counts, not just a change of file: dropping blown
         // frames leaves holes of 8 to 13 s inside a capture, and the Sun has
         // moved by the far side of one just as it has across a file boundary.
         let jumped = (fr.file != dissolveFile)
                   || (fr.index - dissolveIndex > GAP_FRAMES);
         if ( jumped )
         {
            if ( dissolveFile !== null && prevOut !== null && DISSOLVE > 0 )
            {
               dissolveLeft = DISSOLVE;
               dissolveSrc = prevOut;
            }
            dissolveFile = fr.file;
         }
         dissolveIndex = fr.index;
         if ( dissolveLeft > 0 && dissolveSrc !== null )
         {
            let wOld = dissolveLeft/(DISSOLVE + 1);
            for ( let c = 0; c < 3; ++c )
            {
               let a = outPlanes[c], b = dissolveSrc[c];
               for ( let p = 0, n = outW2*outH2; p < n; ++p )
                  a[p] = a[p]*(1 - wOld) + b[p]*wOld;
            }
            --dissolveLeft;
         }
         /*
          * Remember the main view WITHOUT the panels, then draw them.
          *
          * Cross-fading the panels turns their box outlines and leader lines into
          * doubled ghost lines for the length of the dissolve - thin white lines
          * blended against thin white lines somewhere else. The main view should
          * dissolve; the annotation on top of it should not. So the dissolve
          * source is a copy taken before the panels go on, and the panels are
          * composited after, always crisp.
          */
         prevOut = [ outPlanes[0].slice(), outPlanes[1].slice(),
                     outPlanes[2].slice() ];

         /*
          * The panels are drawn at FULL strength on every frame, always.
          *
          * Two cleverer schemes were tried across a dissolve and both were worse
          * than the thing they fixed. Sliding each box from its old position to
          * its new one puts it partway between two eclipse phases, where it points
          * at neither - and drawInsets samples the PANEL at that same position, so
          * every zoom showed the wrong patch of sky while it drifted. Fading the
          * whole annotation in on the dissolve weight avoided that, but a panel
          * whose brightness ramps over three frames reads as the corners flashing.
          *
          * So: constant opacity, correct position, every frame. What is left is
          * that a box reaches its new place up to three frames (0.1 s) before the
          * crescent underneath finishes arriving. That is the smallest of the
          * three artifacts and the only one that never draws anything untrue.
          */
         if ( fr.insets && fr.insets.length )
            drawInsets( outPlanes, fine, DRIZZLE*w, DRIZZLE*h,
                        DRIZZLE*ox, DRIZZLE*oy, outW2, outH2, fr.insets,
                        cfg.insetPanel || 420, cfg.insetZoom || 3 );

         let win = new ImageWindow( outW2, outH2, 3, 8, false, true, "tl" );
         let v = win.mainView;
         v.beginProcess( UndoFlag.NoSwapFile );
         for ( let c = 0; c < 3; ++c )
         {
            v.image.selectedChannel = c;
            v.image.setSamples( outPlanes[c] );
         }
         v.image.resetSelections();
         v.image.apply( GAMMA, ImageOp.Pow );
         v.endProcess();

         // Pin the sample format before saving. Without it PixInsight warns
         // "Insufficient data storage capabilities: PNG" on every frame and
         // converts on the way out, writing 8-bit RGBA rather than the plain
         // 8-bit RGB asked for - an alpha channel nothing here wants, and a
         // silent format conversion per frame.
         win.setSampleFormat( 8, false );
         // Number by the frame's own sequence when the driver supplied one. Under
         // parallel sharding a frame's position within its shard is not its
         // position in the video, and numbering by the loop index would make
         // every shard overwrite the others' output.
         let seq = (fr.seq !== undefined) ? fr.seq : k;
         let name = outDir + "/seq_" + pad( seq ) + ".png";
         if ( !win.saveAs( name, false, false, false, false ) )
            throw new Error( "save failed: " + name );
         win.forceClose();

         if ( (k + 1) % 100 == 0 )
            log( "  rendered " + (k + 1) + "/" + frames.length
                 + " (" + ((Date.now() - t0)/1000/(k + 1)).toFixed( 2 ) + " s/frame)" );
      }
      if ( cur ) cur.close();

      log( "  " + frames.length + " frames in "
           + ((Date.now() - t0)/60000).toFixed( 1 ) + " min" );
      log( "=== TIMELAPSE OK ===" );
   }
   catch ( e )
   {
      log( "*** TIMELAPSE FAILED: " + e.toString() );
      if ( e.stack )
         log( String( e.stack ) );
   }
   logFile.close();
}

/*
 * Read one raw frame and superpixel-demosaic it into R, G, B.
 *
 * RGGB: R at (0,0), G at (1,0) and (0,1), B at (1,1). Superpixel costs no
 * interpolation and, for R and B, throws nothing away either - those channels
 * genuinely are one sample per 2x2 site. Only G is decimated, from a quincunx of
 * two samples down to their mean.
 */
function readSuperpixel( cur, index, W, H, w, h, frameBytes, maxv, R, G, B )
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
         R[o + x] = s[r0 + c]*inv;
         G[o + x] = (s[r0 + c + 1] + s[r1 + c])*0.5*inv;
         B[o + x] = s[r1 + c + 1]*inv;
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

/*
 * Place one plane onto the fine grid, shifted by (tx, ty) plane pixels.
 *
 * With translation-only alignment this is drizzle at pixfrac 1: upsample by the
 * scale factor, offset by scale*shift, add. Dithered sub-pixel shifts across the
 * group fill the fine grid and recover detail past the native sampling. The
 * upsample is bicubic rather than nearest because the dither here is a smooth
 * drift, not a random walk - 20 frames land in only 8 or 9 of the 16 sub-pixel
 * cells, and nearest-neighbour on that leaves visible 2x2 stair-stepping in the
 * cells nothing reached.
 *
 * Passing acc = null returns a fresh accumulator holding just this plane.
 */
function accumulate( arr, w, h, tx, ty, acc )
{
   let up = toImage( arr, w, h );
   up.interpolation = INTERP;
   up.resample( DRIZZLE );
   if ( tx || ty )
   {
      up.interpolation = INTERP;
      up.translate( DRIZZLE*tx, DRIZZLE*ty );
   }
   if ( acc === null )
      return up;
   acc.apply( up, ImageOp.Add );
   up.free();
   return acc;
}

/*
 * Zoomed panels in the corners, with a box and leader lines showing where each
 * one comes from.
 *
 * The panels are sampled from the FINE grid rather than from the cropped output,
 * so a box near the edge of the crop still gets its full surroundings, and the
 * magnification is honest: at zoom 3 one source pixel becomes a 3x3 block, and
 * the detail inside it is whatever the drizzle recovered. Sampling is bilinear -
 * nearest would show the drizzle grid as visible stair-stepping at this
 * magnification, which reads as detail that is not there.
 *
 * There is NOT always one panel per corner. Each inset names its own corner and
 * carries the name of the thing it follows, and gen_insets.py emits only the
 * features that exist in that frame: before first contact there are no cusps, the
 * sunspot spends part of the eclipse behind the Moon, and a totality level may
 * show fewer than four prominences worth pointing at. A panel held on a feature
 * that is not there shows blank sky with a box around it, which is worse than no
 * panel at all.
 */
function drawInsets( outPlanes, fine, fw, fh, ox2, oy2, outW2, outH2, insets,
                     panel, zoom )
{
   const M = 24;                       // margin from the frame edge
   const EDGE = 0.92;                  // line and border brightness
   const corners = [ [ M, M ], [ M, outH2 - panel - M ],
                     [ outW2 - panel - M, M ],
                     [ outW2 - panel - M, outH2 - panel - M ] ];

   function setPx( x, y, v )
   {
      if ( x < 0 || y < 0 || x >= outW2 || y >= outH2 )
         return;
      let i = y*outW2 + x;
      for ( let c = 0; c < 3; ++c )
         outPlanes[c][i] = v;
   }

   function line( x0, y0, x1, y1 )
   {
      let n = Math.ceil( Math.max( Math.abs( x1 - x0 ), Math.abs( y1 - y0 ) ) );
      for ( let s = 0; s <= n; ++s )
      {
         let x = Math.round( x0 + (x1 - x0)*s/n );
         let y = Math.round( y0 + (y1 - y0)*s/n );
         setPx( x, y, EDGE );
         setPx( x + 1, y, EDGE );
      }
   }

   function rect( x0, y0, x1, y1 )
   {
      line( x0, y0, x1, y0 );
      line( x1, y0, x1, y1 );
      line( x1, y1, x0, y1 );
      line( x0, y1, x0, y0 );
   }

   for ( let k = 0; k < insets.length && k < 4; ++k )
   {
      // Source centre: superpixel coordinates, same units as the disc track.
      let scx = DRIZZLE*insets[k].cx, scy = DRIZZLE*insets[k].cy;
      let corner = (insets[k].corner !== undefined) ? insets[k].corner : k;
      let px = corners[corner][0], py = corners[corner][1];
      // Each inset may magnify at its own rate. A cusp is a needle and needs a
      // much tighter box than a sunspot group or a prominence.
      let z = insets[k].zoom || zoom;
      let hb = panel/z/2;              // half the source box, fine px

      for ( let v = 0; v < panel; ++v )
      {
         let sy = scy - hb + v/z;
         let y0 = Math.floor( sy ), fy = sy - y0;
         for ( let u = 0; u < panel; ++u )
         {
            let sx = scx - hb + u/z;
            let x0 = Math.floor( sx ), fx = sx - x0;
            let o = (py + v)*outW2 + px + u;
            if ( x0 < 0 || y0 < 0 || x0 + 1 >= fw || y0 + 1 >= fh )
            {
               for ( let c = 0; c < 3; ++c )
                  outPlanes[c][o] = 0;
               continue;
            }
            let a = y0*fw + x0;
            for ( let c = 0; c < 3; ++c )
            {
               let p = fine[c];
               outPlanes[c][o] =
                  (p[a]*(1 - fx) + p[a + 1]*fx)*(1 - fy)
                + (p[a + fw]*(1 - fx) + p[a + fw + 1]*fx)*fy;
            }
         }
      }

      // The source box, in output coordinates.
      let bx0 = scx - hb - ox2, by0 = scy - hb - oy2;
      let bx1 = scx + hb - ox2, by1 = scy + hb - oy2;
      rect( bx0, by0, bx1, by1 );
      rect( px, py, px + panel - 1, py + panel - 1 );

      /*
       * Leader lines from the box to the panel.
       *
       * Join the two box corners on the side FACING the panel to the two panel
       * corners facing the box. Choosing the near side per panel rather than a
       * fixed pair is what stops the lines crossing the frame diagonally when a
       * box happens to drift past the panel it belongs to - which the cusps do,
       * since they travel right across the disc over the course of the eclipse.
       */
      let panelLeft = (px + panel/2) < (bx0 + bx1)/2;
      let sbx = panelLeft ? bx0 : bx1;
      let spx = panelLeft ? px + panel - 1 : px;
      line( sbx, by0, spx, py );
      line( sbx, by1, spx, py + panel - 1 );

      /*
       * Name the subject, OUTSIDE the panel.
       *
       * Four magnified squares with nothing said about them make the viewer guess
       * which is which, and two of the five subjects here - the deepest incursion
       * of the lunar limb, and a cusp - are not self-evident even to someone who
       * knows the eclipse. The label goes below a top panel and above a bottom one
       * so it never covers the magnified view, and it sits on a darkened plate
       * because it otherwise has to be legible against both black sky and the
       * white photosphere.
       */
      if ( insets[k].label )
      {
         let top = (py < outH2/2);
         let scale = textScale( insets[k].label, panel );
         let tw = textWidth( insets[k].label, scale );
         let ty = top ? py + panel + 3*scale : py - 7*scale - 3*scale;
         let tx = (px + panel/2 < outW2/2) ? px : px + panel - tw;
         text( insets[k].label, tx, ty, scale );
      }
   }

   /* Largest scale at which the label still fits the panel it belongs to. */
   function textScale( s, width )
   {
      for ( let z = 5; z > 1; --z )
         if ( textWidth( s, z ) <= width )
            return z;
      return 1;
   }

   function textWidth( s, z )
   {
      return s.length > 0 ? (6*s.length - 1)*z : 0;
   }

   function text( s, x, y, z )
   {
      s = String( s ).toUpperCase();
      let tw = textWidth( s, z ), th = 7*z, pad = 2*z;

      // Darkened plate. Multiplying rather than filling keeps it invisible
      // against the black sky, where a solid box would be a grey rectangle.
      for ( let v = -pad; v < th + pad; ++v )
      {
         let yy = y + v;
         if ( yy < 0 || yy >= outH2 )
            continue;
         for ( let u = -pad; u < tw + pad; ++u )
         {
            let xx = x + u;
            if ( xx < 0 || xx >= outW2 )
               continue;
            let i = yy*outW2 + xx;
            for ( let c = 0; c < 3; ++c )
               outPlanes[c][i] *= 0.25;
         }
      }

      for ( let j = 0; j < s.length; ++j )
      {
         let g = FONT[s.charAt( j )];
         if ( !g )
            continue;
         let gx = x + 6*j*z;
         for ( let r = 0; r < 7; ++r )
            for ( let b = 0; b < 5; ++b )
            {
               if ( !(g[r] & (1 << (4 - b))) )
                  continue;
               for ( let dy = 0; dy < z; ++dy )
                  for ( let dx = 0; dx < z; ++dx )
                     setPx( gx + b*z + dx, y + r*z + dy, EDGE );
            }
      }
   }
}

/*
 * A 5x7 bitmap font, drawn a pixel at a time.
 *
 * PJSR's Graphics/Bitmap classes are not reachable from this script - it works on
 * plain Float32Array planes and never builds an ImageWindow until the frame is
 * finished - so the labels are rasterised here. Each glyph is seven rows of five
 * bits, most significant bit leftmost.
 */
var FONT = {
   " ": [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ],
   "-": [ 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00 ],
   ".": [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C ],
   "A": [ 0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 ],
   "B": [ 0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E ],
   "C": [ 0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E ],
   "D": [ 0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E ],
   "E": [ 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F ],
   "F": [ 0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10 ],
   "G": [ 0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F ],
   "H": [ 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 ],
   "I": [ 0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E ],
   "J": [ 0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C ],
   "K": [ 0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11 ],
   "L": [ 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F ],
   "M": [ 0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11 ],
   "N": [ 0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11 ],
   "O": [ 0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E ],
   "P": [ 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10 ],
   "Q": [ 0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D ],
   "R": [ 0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11 ],
   "S": [ 0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E ],
   "T": [ 0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04 ],
   "U": [ 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E ],
   "V": [ 0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04 ],
   "W": [ 0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11 ],
   "X": [ 0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11 ],
   "Y": [ 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04 ],
   "Z": [ 0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F ],
   "0": [ 0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E ],
   "1": [ 0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E ],
   "2": [ 0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F ],
   "3": [ 0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E ],
   "4": [ 0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02 ],
   "5": [ 0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E ],
   "6": [ 0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E ],
   "7": [ 0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08 ],
   "8": [ 0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E ],
   "9": [ 0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C ]
};

function pad( i )
{
   let s = String( i );
   while ( s.length < 5 )
      s = "0" + s;
   return s;
}

/* Intensity-weighted centroid of the brightest pixels — the partial crescent, or
 * the corona once the filter is off. Both are centred on the Sun. */
function centroid( g, w, h )
{
   let n = w*h;
   let lo = Infinity, hi = -Infinity;
   for ( let i = 0; i < n; ++i )
   {
      let v = g[i];
      if ( v < lo ) lo = v;
      if ( v > hi ) hi = v;
   }
   const BINS = 1024;
   let hist = new Int32Array( BINS );
   let scale = (hi > lo) ? (BINS - 1)/(hi - lo) : 0;
   for ( let i = 0; i < n; ++i )
      hist[Math.round( (g[i] - lo)*scale )]++;
   let want = Math.max( 1, Math.round( CENTROID_FRACTION*n ) );
   let acc = 0, bin = BINS - 1;
   for ( ; bin > 0 && acc < want; --bin )
      acc += hist[bin];
   let thr = lo + bin/(scale || 1);

   let sw = 0, sx = 0, sy = 0;
   for ( let y = 0; y < h; ++y )
   {
      let row = y*w;
      for ( let x = 0; x < w; ++x )
      {
         let d = g[row + x] - thr;
         if ( d > 0 ) { sw += d; sx += d*x; sy += d*y; }
      }
   }
   if ( sw <= 0 )
      return { cx: w/2, cy: h/2 };
   return { cx: sx/sw, cy: sy/sw };
}

function clamp( v, lo, hi )
{
   return v < lo ? lo : (v > hi ? hi : v);
}

/* Bilinear sub-pixel crop and linear gain in one pass. */
function cropGain( src, w, h, ox, oy, outW, outH, gain )
{
   let dst = new Float32Array( outW*outH );
   for ( let y = 0; y < outH; ++y )
   {
      let sy = oy + y;
      let y0 = Math.floor( sy ), fy = sy - y0, y1 = y0 + 1;
      // Outside the sensor, leave black. Clamping to the edge pixel would smear
      // a bright streak along the border and pretend it was data.
      if ( y0 < 0 || y1 >= h ) continue;
      let r0 = y0*w, r1 = y1*w, drow = y*outW;
      for ( let x = 0; x < outW; ++x )
      {
         let sx = ox + x;
         let x0 = Math.floor( sx ), fx = sx - x0, x1 = x0 + 1;
         if ( x0 < 0 || x1 >= w ) continue;
         let a = src[r0 + x0] + (src[r0 + x1] - src[r0 + x0])*fx;
         let b = src[r1 + x0] + (src[r1 + x1] - src[r1 + x0])*fx;
         let v = (a + (b - a)*fy)*gain;
         if ( v > SHOULDER_KNEE )
         {
            let span = SHOULDER_CEIL - SHOULDER_KNEE;
            v = SHOULDER_KNEE + span*Math.tanh( (v - SHOULDER_KNEE)/span );
         }
         dst[drow + x] = v < 0 ? 0 : v;
      }
   }
   return dst;
}

main();
