
# 定义特定后缀名的数组
$supportedExtensions = @(".py", ".dll", ".pyd", ".ps1")

# 获取当前目录下所有文件
$files = Get-ChildItem -File

# 遍历文件并删除不支持的后缀名的文件
foreach ($file in $files) {
    if ($supportedExtensions -notcontains $file.Extension) {
        # 删除文件
        Remove-Item $file.FullName -Force
        Write-Host "del file: $($file.Name)"
    }
}