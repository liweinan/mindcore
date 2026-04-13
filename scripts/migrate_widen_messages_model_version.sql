-- 已有库：ollama:<模型名> 可超过 32 字符，执行一次即可。
ALTER TABLE messages ALTER COLUMN model_version TYPE VARCHAR(128);
