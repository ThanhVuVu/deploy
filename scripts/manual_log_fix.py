
import json
import os
from datetime import datetime, timezone, timedelta

VN_TZ = timezone(timedelta(hours=7))

input_path = '.ai-log/session.jsonl'
log_path = r'C:\Users\phanc\.gemini\antigravity\brain\8eb08875-ac78-41b7-9d6f-020a2c5bb573\.system_generated\logs\overview.txt'
output_path = '.ai-log/session.jsonl'

# 1. Collect all valid entries from existing file (only the non-antigravity ones, to be safe, or anything that parses)
valid_entries = []
if os.path.exists(input_path):
    with open(input_path, 'rb') as f:
        for line in f:
            # Clean nulls and try decode
            line = line.replace(b'\x00', b'')
            for enc in ['utf-8', 'utf-16', 'cp1252']:
                try:
                    decoded = line.decode(enc).strip()
                    if not decoded: continue
                    entry = json.loads(decoded)
                    # Skip antigravity entries from the current session because we will re-generate them correctly
                    if entry.get('tool') == 'antigravity' and entry.get('session_id') == '8eb08875-ac78-41b7-9d6f-020a2c5bb573':
                        continue
                    valid_entries.append(entry)
                    break
                except:
                    continue

# 2. Extract ALL prompts from overview.txt
session_id = '8eb08875-ac78-41b7-9d6f-020a2c5bb573'
student = 'thang.nt225530@sis.hust.edu.vn'
repo = 'A20-App-115'
branch = 'rag'
commit = 'c603f44'

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('type') == 'USER_INPUT' and data.get('source') == 'USER_EXPLICIT':
                prompt = data.get('content', '')
                if '<USER_REQUEST>' in prompt:
                    prompt = prompt.split('<USER_REQUEST>')[1].split('</USER_REQUEST>')[0].strip()
                
                ts_utc = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
                ts_vn = ts_utc.astimezone(VN_TZ).isoformat()
                
                entry = {
                    "ts": ts_vn,
                    "tool": "antigravity",
                    "event": "BeforeAgent",
                    "session_id": session_id,
                    "model": "",
                    "repo": repo,
                    "branch": branch,
                    "commit": commit,
                    "student": student,
                    "prompt": prompt
                }
                valid_entries.append(entry)
        except:
            continue

# 3. Write back everything as clean UTF-8
with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
    for entry in valid_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Total entries saved: {len(valid_entries)}")
