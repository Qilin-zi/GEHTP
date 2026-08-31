# ===== 一键脚本: 在 Windows 上跑 Stage 4-5 (部署+推理) =====
# 机器: 本地 Windows
# 产出: 板子上的文本生成
#
# 用法 (PowerShell):
#   .\run_on_windows.ps1
#   .\run_on_windows.ps1 "你的提示词"
#
# 前置:
#   - run_on_dev133.sh 已完成 (v73 .bin 在 dev133 上)
#   - adb 可连接板子
#   - plink 可连接 dev133

$ErrorActionPreference = "Stop"

# ===== 配置 =====
$DEV133_HOST = "10.137.185.133"
$DEV133_USER = "rqilin"
$DEV133_PASS = "Wangba521."
$DEV133_FP = "SHA256:yqoUk1QlZnVDwokLgXrGT8gdoGyHnTcOLL/Bi6aiF2c"

$ADB = "C:\Users\RQILIN\AppData\Local\Programs\platform-tools\adb.exe"
$PLINK = "C:\Users\RQILIN\AppData\Local\Temp\opencode\plink.exe"

$DEV133_BIN_DIR = "/data01/rqilin/qwen3_llm_v2/example2/host_linux/assets/artifacts/ar128-cl2048_v73"
$BOARD_MODEL_DIR = "/data/qairt/qwen3"
$BOARD_BIN_DIR = "/data/qairt/qwen3/serialized_v73"
$LOCAL_TMP = "C:\Users\RQILIN\AppData\Local\Temp\opencode\qwen3_transfer"

$NUM_SPLITS = 3
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "############################################################"
Write-Host "# Qwen3 QNN HTP v73 部署+推理 (Windows)                      #"
Write-Host "############################################################"

# ===== 工具函数: 用 plink 下载二进制文件 (流重定向,避免编码问题) =====
function Download-File {
    param([string]$Remote, [string]$Local)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $PLINK
    $psi.Arguments = "-ssh -batch -pw $DEV133_PASS -hostkey $DEV133_FP $DEV133_USER@$DEV133_HOST `"cat $Remote`""
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    $stream = [System.IO.File]::Create($Local)
    $p.StandardOutput.BaseStream.CopyTo($stream)
    $stream.Close()
    $p.WaitForExit()
    if ($p.ExitCode -ne 0) {
        $err = $p.StandardError.ReadToEnd()
        Write-Host "  [错误] plink 退出码 $($p.ExitCode): $err"
    }
}

# ===== Step 1: 从 dev133 下载 .bin =====
Write-Host "`n===== [1/4] 从 dev133 下载 v73 .bin ====="
New-Item -ItemType Directory -Path $LOCAL_TMP -Force | Out-Null

for ($i = 1; $i -le $NUM_SPLITS; $i++) {
    $binFile = "ar128-cl2048_${i}_of_${NUM_SPLITS}_v73.serialized.bin"
    $remotePath = "$DEV133_BIN_DIR/$binFile"
    $localPath = "$LOCAL_TMP\${i}_v73.bin"

    Write-Host "  下载 ${i}/${NUM_SPLITS}..."
    Download-File -Remote $remotePath -Local $localPath

    $size = (Get-Item $localPath).Length
    Write-Host "    本地: $size bytes ($([math]::Round($size/1MB,0)) MB)"
}

# ===== Step 2: adb push 到板子 =====
Write-Host "`n===== [2/4] adb push 到板子 ====="
& $ADB root 2>$null; Start-Sleep 2; & $ADB wait-for-device
& $ADB shell "mkdir -p $BOARD_BIN_DIR"

# 清理旧文件
& $ADB shell "rm -f $BOARD_BIN_DIR/*.bin $BOARD_BIN_DIR/*.bin.bin 2>/dev/null"

for ($i = 1; $i -le $NUM_SPLITS; $i++) {
    $localPath = "$LOCAL_TMP\${i}_v73.bin"
    Write-Host "  push ${i}/${NUM_SPLITS}..."
    & $ADB push $localPath "$BOARD_BIN_DIR/"
}

# ===== Step 3: 推送 GenieX 配置 =====
Write-Host "`n===== [3/4] 推送配置 ====="
& $ADB push "$SCRIPT_DIR\stage4_deploy\genie_qwen3_v73.json" "$BOARD_MODEL_DIR/"

# Tokenizer (如板子上没有)
$hasTok = & $ADB shell "ls $BOARD_MODEL_DIR/tokenizer.json 2>/dev/null"
if (-not $hasTok) {
    $localTok = "$SCRIPT_DIR\stage4_deploy\tokenizer.json"
    if (Test-Path $localTok) {
        & $ADB push $localTok "$BOARD_MODEL_DIR/"
    } else {
        # 从 dev133 拉 tokenizer
        Write-Host "  从 dev133 拉 tokenizer..."
        $tokLocal = "$LOCAL_TMP\tokenizer.json"
        Download-File -Remote "/data01/rqilin/qwen3_llm_v2/example1/output_dir_/tokenizer/tokenizer.json" -Local $tokLocal
        & $ADB push $tokLocal "$BOARD_MODEL_DIR/"
    }
}

# 验证
Write-Host "`n===== 验证 ====="
Write-Host "板子 .bin 文件:"
& $ADB shell "ls -lh $BOARD_BIN_DIR/"
Write-Host "`n板子配置文件:"
& $ADB shell "ls -lh $BOARD_MODEL_DIR/genie_qwen3_v73.json $BOARD_MODEL_DIR/tokenizer.json 2>/dev/null"

# ===== Step 4: 推理 =====
Write-Host "`n===== [4/4] 推理 ====="
$PROMPT = if ($args.Count -gt 0) { $args[0] } else { "Hello, how are you?" }
Write-Host "Prompt: $PROMPT`n"

& $ADB shell ". /data/qairt/env.sh && cd $BOARD_MODEL_DIR && genie-t2t-run -c genie_qwen3_v73.json -p '$PROMPT' 2>&1"

# 清理本地临时文件
Write-Host "`n===== 清理 ====="
Remove-Item "$LOCAL_TMP\*_v73.bin" -ErrorAction SilentlyContinue
Remove-Item "$LOCAL_TMP\tokenizer.json" -ErrorAction SilentlyContinue
Remove-Item $LOCAL_TMP -ErrorAction SilentlyContinue

Write-Host "`n############################################################"
Write-Host "# 全流程完成!                                                #"
Write-Host "############################################################"
