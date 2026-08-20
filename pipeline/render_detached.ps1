<#
    Render and encode in a DETACHED process, so the render outlives whatever
    started it.

    A long render on this machine kept dying part-way - six times in one day, at
    10 to 21 minutes, at different frame counts. It is not memory: 98 GB free
    during a render, 15 GB of private bytes across 26 workers, no
    Resource-Exhaustion events. It is not Windows killing python either - four of
    the five kills left ZERO events in the System or Application logs within 90
    seconds, and a process the OS kills leaves a trace.

    What dies is the shell the render is a child of. So do not be a child of it:
    Start-Process detaches, the launcher returns immediately, and the render
    keeps going with its output in a log. Verified - a detached process kept
    writing across the boundary of the task that spawned it.

    Poll the log, or ecl.progress, for completion.
#>
param(
    [string]$Cfg    = "S:\solar-eclipse\out\configs\timelapse.json",
    [string]$Data   = "Z:\solar-eclipse\Sun",
    [string]$Frames = "Z:\eclipse-work\tl_py",
    [string]$Final  = "Z:\eclipse-work\final",
    [string]$Log    = "Z:\eclipse-work\render.log",
    [switch]$Resume,          # keep what is on disk and fill the gaps
    [switch]$KeepPrevious     # do not park the existing cut
)

$py = "S:\solar-eclipse\pipeline\.venv\Scripts\python.exe"

if (-not $Resume -and -not $KeepPrevious) {
    $st = Get-Date -Format "yyyyMMdd-HHmm"
    New-Item -ItemType Directory -Force "$Frames`_$st" | Out-Null
    Get-ChildItem "$Frames\seq_*.png" -ErrorAction SilentlyContinue |
        Move-Item -Destination "$Frames`_$st\"
    New-Item -ItemType Directory -Force "$Final`_$st" | Out-Null
    Get-ChildItem "$Final\*.mp4" -ErrorAction SilentlyContinue |
        Move-Item -Destination "$Final`_$st\"
    Write-Host "parked the previous cut as *_$st"
}

# The expected count comes from the config, never a literal: it changes whenever
# a dwell does, and a stale literal fails a good render.
$want = (Get-Content $Cfg -Raw | ConvertFrom-Json).frames.Count
$resumeArg = if ($Resume) { "--resume" } else { "" }

$inner = @"
`$ErrorActionPreference = 'Stop'
"[render] `$(Get-Date -Format HH:mm:ss)" | Out-File -FilePath '$Log' -Encoding utf8
& '$py' -m ecl.tl_render --config '$Cfg' --data-dir '$Data' --out-dir '$Frames' $resumeArg *>> '$Log'
if (`$LASTEXITCODE -ne 0) { "render FAILED `$LASTEXITCODE" | Add-Content '$Log'; exit 1 }
`$n = (Get-ChildItem '$Frames\seq_*.png').Count
if (`$n -ne $want) { "expected $want frames, have `$n" | Add-Content '$Log'; exit 1 }
"[encode] `$(Get-Date -Format HH:mm:ss)" | Add-Content '$Log'
& '$py' -m ecl.encode --config '$Cfg' --frames '$Frames' --out-dir '$Final' *>> '$Log'
if (`$LASTEXITCODE -ne 0) { "encode FAILED `$LASTEXITCODE" | Add-Content '$Log'; exit 1 }
"ALL DONE `$(Get-Date -Format HH:mm:ss)" | Add-Content '$Log'
"@

Set-Location "S:\solar-eclipse\pipeline"
$p = Start-Process powershell `
        -ArgumentList "-NoProfile", "-WindowStyle", "Hidden", "-Command", $inner `
        -WorkingDirectory "S:\solar-eclipse\pipeline" -PassThru

Write-Host "detached render pid $($p.Id), $want frames expected"
Write-Host "  log:      $Log"
Write-Host "  progress: $py -m ecl.progress --frames `"$Frames`" --watch"
