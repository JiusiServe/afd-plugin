import io

path = "afd-plugin/recipe/npu/CAMAsyncAFDConnector/deepseek_v4/attention_tp8.sh"
with io.open(path, "r", encoding="utf-8") as f:
    text = f.read()
old = "  --enable-expert-parallel \\\n  --enable-dbo \\\n"
new = "  --enable-expert-parallel \\\n"
assert old in text, "enable-dbo block not found"
text = text.replace(old, new)
with io.open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
for i, line in enumerate(text.splitlines(), 1):
    if "--enable-dbo" in line or "async_moe_ubatching" in line:
        print(i, line)
print("done")
