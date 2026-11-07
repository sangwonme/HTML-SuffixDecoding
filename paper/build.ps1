# Rebuild GenUI_claims_KO_EN.pdf (Korean first, then English). Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File paper\build.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$opts = @('--enable-installer', '-interaction=nonstopmode', '-halt-on-error')
foreach ($f in 'main_ko', 'main_en') {
    foreach ($pass in 1, 2) {   # two passes for refs
        & xelatex @opts "$f.tex" | Out-Null
        if ($LASTEXITCODE -ne 0) { Get-Content "$f.log" -Tail 30; throw "xelatex failed on $f" }
    }
}
& xelatex @opts merge.tex | Out-Null
if ($LASTEXITCODE -ne 0) { Get-Content merge.log -Tail 30; throw 'merge failed' }
Copy-Item merge.pdf GenUI_claims_KO_EN.pdf -Force
foreach ($f in 'main_ko', 'main_en', 'merge') {
    $m = Select-String -Path "$f.log" -Pattern 'Output written on .* \((\d+) page'
    Write-Host "$f : $($m.Matches[0].Groups[1].Value) pages"
}
Write-Host "OK -> $PSScriptRoot\GenUI_claims_KO_EN.pdf"
