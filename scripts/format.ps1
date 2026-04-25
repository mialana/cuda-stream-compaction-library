Get-ChildItem -Recurse -Path tests, stream_compaction `
    -Include *.h, *.cpp, *.inl, *.cu | 
    ForEach-Object { clang-format -i $_.FullName}
