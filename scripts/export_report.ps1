param([string]$ReportDir = (Join-Path $PSScriptRoot '..\report'))
$ErrorActionPreference = 'Stop'
$resolved = (Resolve-Path -LiteralPath $ReportDir).Path
$stems = @('bao_cao_do_an_ai4i', 'tom_tat_bao_ve_1_trang')
foreach ($stem in $stems) {
    $word = $null
    try {
        $word = New-Object -ComObject Word.Application
        $word.Visible = $false
        $word.DisplayAlerts = 0
        $doc = $null
        try {
            $doc = $word.Documents.Open((Join-Path $resolved ($stem + '.docx')), $false, $true)
            $doc.Repaginate()
            $doc.ExportAsFixedFormat((Join-Path $resolved ($stem + '.pdf')), 17)
            Write-Output ($stem + ': ' + $doc.ComputeStatistics(2) + ' pages')
        } finally {
            if ($null -ne $doc) {
                try { $doc.Close(0) } catch { Write-Warning $_.Exception.Message }
                [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
            }
        }
    } finally {
        if ($null -ne $word) {
            try { $word.Quit() } catch { Write-Warning $_.Exception.Message }
            [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
        }
    }
}
