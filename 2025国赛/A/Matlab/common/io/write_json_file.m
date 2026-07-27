function write_json_file(filePath, value)
%WRITE_JSON_FILE 以 UTF-8 写入结构化 JSON 文件。

jsonText = jsonencode(value, PrettyPrint=true);
[fid, message] = fopen(filePath, 'w', 'n', 'UTF-8');
if fid < 0
    error('SmokeOutput:JsonWriteFailed', '无法打开 JSON 文件 %s：%s', filePath, message);
end
cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s', jsonText);
end
