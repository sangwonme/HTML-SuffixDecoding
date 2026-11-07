# Build genui_paper.pdf (English research paper draft). Run:
#   powershell -ExecutionPolicy Bypass -File paper\build_paper.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$opts = @('--enable-installer', '-interaction=nonstopmode', '-halt-on-error')
foreach ($pass in 1, 2) {
    & xelatex @opts genui_paper.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { Get-Content genui_paper.log -Tail 30; throw 'xelatex failed' }
}
$w = Select-String -Path genui_paper.log -Pattern '^LaTeX Warning|^Package .* Warning|Overfull|Underfull'
if ($w) { $w | ForEach-Object { Write-Host $_.Line } } else { Write-Host 'no warnings' }
$m = Select-String -Path genui_paper.log -Pattern 'Output written on .* \((\d+) page'
Write-Host "genui_paper.pdf : $($m.Matches[0].Groups[1].Value) pages"
