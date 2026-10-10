"""Display names and exact scope of the frozen benchmark snapshots."""
LABELS=dict(main='Upstream main',optimistic='Earlier five-PR subset',proposed6='Currently proposed changes (6 PRs)',
            cpu_stack='CPU experimental (no verification)',gpu='GPU experimental (no verification)')
SCOPE=('Currently proposed changes = pinned upstream main `ea1a15c` plus performance '
       'PRs [#4](https://github.com/navinlabcode/Copykat_python/pull/4), '
       '[#5](https://github.com/navinlabcode/Copykat_python/pull/5), '
       '[#6](https://github.com/navinlabcode/Copykat_python/pull/6), '
       '[#7](https://github.com/navinlabcode/Copykat_python/pull/7), '
       '[#8](https://github.com/navinlabcode/Copykat_python/pull/8) and '
       '[#9](https://github.com/navinlabcode/Copykat_python/pull/9). '
       'The new six-PR filesystem snapshot includes the Arrow writer; its fresh measurements '
       'run serially after accuracy completes. The original five-PR snapshot '
       'of PRs #4 and #6–#9 excluded #5; its evidence is retained in the historical archives. '
       'The open CLI fix #3 is outside both performance snapshots. CPU experimental '
       '(no verification) = `perf/exact-shortcuts` at `ff63f19`, with Arrow output, memory/storage '
       'refactors, Ward-engine and repeated-bin changes. This broader experimental '
       'integration is distinct from the open-PR snapshots. Source pins and patch order '
       'are recorded; no Git branch or ref was changed.')
