#!/usr/bin/env python3
"""Read-only Tkinter dashboard for a local PoSSUM frozen-expert run."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import time
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}  # A writer may be between truncation and completion.


def duration(seconds):
    if seconds is None:
        return '—'
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}h {m:02d}m' if h else f'{m}m {s:02d}s'


def snapshot(run):
    log = run / 'pipeline.log'
    try:
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 64000))
            text = f.read().decode('utf-8', errors='replace')
        age = time.time() - log.stat().st_mtime
    except OSError:
        text, age = '', None
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text).replace('\r', '\n')
    split = read_json(run / 'split.json').get('pilot', {})
    total = sum(len(ids) for ids in split.values())
    compact = read_json(run / 'report.json')
    final = read_json(run / 'four-expert-report.json')
    rows = []
    for name in ('bge', 'bert', 'gliner', 'mistral', 'qwen'):
        progress = read_json(run / f'{name}_progress.json')
        done = name in compact.get('runtime_seconds', {})
        done |= name in ('bge', 'bert') and (run / f'{name}_features.npy').exists()
        done |= name == 'gliner' and (run / 'gliner_raw.json').exists()
        active = f'Running {name} on' in text
        if name in ('mistral', 'qwen'):
            active = f'Downloading/loading mlx-community/{"Mistral" if name == "mistral" else "Qwen"}' in text
            done = (run / f'{name}_raw.json').exists()
        count = progress.get('completed', total if done else 0)
        n = progress.get('total', total)
        if name == 'gliner' and not done:
            matches = re.findall(r'GLiNER: (\d+)/(\d+) passages', text)
            if matches:
                count, n = map(int, matches[-1])
        state = 'Complete' if done else 'Scoring' if count else 'Loading / running' if active else 'Queued'
        speed = progress.get('seconds_per_excerpt')
        eta = (n-count)*speed if speed and not done else 0 if done else None
        elapsed = compact.get('runtime_seconds', {}).get(name)
        if progress:
            elapsed = sum(progress.get('chunk_seconds', []))
        rows.append((name.upper(), state, f'{count:,} / {n:,}' if n else '—',
                     f'{count/n:.0%}' if n else '—', duration(elapsed),
                     f'{speed:.2f} s' if speed else '—', duration(eta)))
    rows.append(('ROUTER + EVAL', 'Complete' if final else 'Queued', '3 seeds' if final else '—',
                 '100%' if final else '—', duration(final.get('routing_training_seconds')), '—', '—'))
    if final:
        stage = 'Complete — four-expert routing evaluation saved'
    elif 'Traceback (most recent call last)' in text:
        stage = 'Error detected — inspect the log below'
    else:
        running = [r for r in rows if r[1] not in ('Queued', 'Complete')]
        stage = f'{running[-1][0]} · {running[-1][1]}' if running else 'Waiting for the next stage'
    return rows, stage, text, age, final


class Monitor:
    def __init__(self, window, run):
        self.window, self.run = window, run
        self.previous = None
        window.title('PoSSUM · Training monitor')
        window.geometry('1080x740')
        window.minsize(850, 560)
        style = ttk.Style(window)
        style.configure('Title.TLabel', font=('Helvetica', 22, 'bold'))
        style.configure('Stage.TLabel', font=('Helvetica', 14))
        style.configure('Treeview', rowheight=32)
        main = ttk.Frame(window, padding=20)
        main.pack(fill='both', expand=True)
        ttk.Label(main, text='PoSSUM', style='Title.TLabel').pack(anchor='w')
        ttk.Label(main, text='Frozen experts · neural routing + self prediction').pack(anchor='w', pady=(2, 12))
        self.stage = ttk.Label(main, style='Stage.TLabel')
        self.stage.pack(anchor='w')
        self.updated = ttk.Label(main)
        self.updated.pack(anchor='w', pady=(5, 12))
        columns = ('component', 'state', 'passages', 'percent', 'elapsed', 'speed', 'eta')
        self.table = ttk.Treeview(main, columns=columns, show='headings', height=6, selectmode='none')
        for key, title, width in zip(columns, ('Component', 'State', 'Scored', 'Progress', 'Scoring time', 'Per passage', 'Stage ETA'), (155, 170, 130, 85, 110, 100, 110)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, minwidth=70)
        self.table.pack(fill='x')
        for i in range(6):
            self.table.insert('', 'end', iid=str(i))
        ttk.Label(main, text='ETA covers the current scoring stage only; downloads, loading and queued models are excluded.').pack(anchor='w', pady=(8, 2))
        ttk.Label(main, text='Refreshes every 2 seconds. No recent log output can mean a download or model load is still running.').pack(anchor='w')
        bar = ttk.Frame(main)
        bar.pack(fill='x', pady=(12, 5))
        ttk.Label(bar, text='Live pipeline log').pack(side='left')
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text='Follow latest output', variable=self.follow).pack(side='right')
        self.log = ScrolledText(main, height=15, wrap='word', font=('Menlo', 11), bg='#142033', fg='#e4edf8', insertbackground='white')
        self.log.pack(fill='both', expand=True)
        self.log.configure(state='disabled')
        ttk.Label(main, text=str(run)).pack(anchor='w', pady=(10, 0))
        self.refresh()

    def refresh(self):
        rows, stage, text, age, final = snapshot(self.run)
        self.stage.configure(text=stage)
        clock = dt.datetime.now().astimezone().strftime('%H:%M:%S %Z')
        self.updated.configure(text=f'Checked {clock} · Last log update {duration(age)} ago' if age is not None else f'Checked {clock} · Waiting for pipeline.log')
        for i, row in enumerate(rows):
            self.table.item(str(i), values=row)
        if text != self.previous:
            position = self.log.yview()[0]
            self.log.configure(state='normal')
            self.log.delete('1.0', 'end')
            self.log.insert('end', text or 'Waiting for run output…')
            self.log.configure(state='disabled')
            self.log.see('end') if self.follow.get() else self.log.yview_moveto(position)
            self.previous = text
        self.window.after(2000, self.refresh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/frozen-four-42')
    parser.add_argument('--smoke-test', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    window = tk.Tk()
    Monitor(window, args.run.resolve())
    if args.smoke_test:
        window.after(800, window.destroy)
    window.mainloop()


if __name__ == '__main__':
    main()
