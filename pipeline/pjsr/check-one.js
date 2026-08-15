/*
 * check-one.js — report why a PJSR script dies on load.
 *
 * In -r mode PixInsight discards script load errors and exits 0, so a syntax or
 * preprocessor problem is indistinguishable from a script that ran and did
 * nothing. This evals a file with its directives stripped and writes the first
 * error, with a line number, to a report file.
 *
 *   -r="...check-one.js,<scriptPath>,<reportPath>"
 */

#engine v8

function main()
{
   let target = jsArguments[0];
   let reportPath = jsArguments[1];
   let out = [];
   function rep( s )
   {
      out.push( String( s ) );
      File.writeTextFile( reportPath, out.join( "\n" ) + "\n" );
   }
   rep( "checker alive: " + target );

   let src = File.readFile( target ).utf8ToString();
   rep( "read " + src.length + " chars" );

   // Strip preprocessor directives; keep line count identical so the reported
   // line number points at the original file.
   let lines = src.split( "\n" );
   for ( let i = 0; i < lines.length; ++i )
      if ( /^\s*#/.test( lines[i] ) )
         lines[i] = "";
   // Do not actually run it — only parse and define.
   let body = lines.join( "\n" ).replace( /\nmain\(\);\s*$/, "\n" );

   try
   {
      eval( body + "\n;null;" );
      rep( "OK: parsed and evaluated cleanly" );
   }
   catch ( e )
   {
      rep( "FAIL: " + e.name + ": " + e.message
           + (e.lineNumber !== undefined ? " (line " + e.lineNumber + ")" : "") );
      if ( e.stack )
         rep( "stack: " + String( e.stack ).split( "\n" ).slice( 0, 4 ).join( " | " ) );
   }
   rep( "CHECK DONE" );
}

main();
