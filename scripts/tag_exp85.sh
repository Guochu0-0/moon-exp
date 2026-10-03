#!/bin/bash
# 给旧实验用过、不在 main 上的 commit 打 tag 并推到 GitHub（#85）。由 scripts/backfill_launch85.py 生成。
# 在本机主 checkout 里运行：其中几个 commit 不在任何分支上，只在本机的 git 对象库里。
set -e
git tag -a exp/B0-23d65aa 23d65aad1220ec99d653e1bee705a5e95fca6e2e -m 'B0 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/B0-4b2a395 4b2a395040ba2cbb5dba07b3462d0666701f3556 -m 'B0 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/B0-563b738 563b7380344287951e1dea843b781505407531e4 -m 'B0 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/B0-6a75499 6a7549987e60cb1cd06ae6a3e24672c55e1f9ff2 -m 'B0 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/B0-6c4e3f3 6c4e3f3827b5c5c4d46ec29fbe60723669e88c8e -m 'B0 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/B0-7224de2 7224de2c0017a407a8d3f1f5939b99206073ec37 -m 'B0 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/B0-cb293a8 cb293a84c2b9f076817f457c6bf0da60df6ea6a9 -m 'B0 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/B0m-7224de2 7224de2c0017a407a8d3f1f5939b99206073ec37 -m 'B0m 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/B0m-880b580 880b580a3fcfc661c5a6f9b69bebe5b74d474b3b -m 'B0m 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/B0m-9e85c61 9e85c61735e110935762998f62da0236bc8d65d6 -m 'B0m 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/B0m-cb293a8 cb293a84c2b9f076817f457c6bf0da60df6ea6a9 -m 'B0m 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/R-0b64893 0b6489343ee2e28cddc50da1683b64b99f70af6b -m 'R 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/R-76687c2 76687c235dbc2867485e53fd16371d1e26f1a75a -m 'R 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/R-e8a8e39 e8a8e39396cd444fa5479e129bf3166762713635 -m 'R 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/R26-00343c1 00343c121a18ec4478cafad53237b97255b1955b -m 'R26 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/R26-ab3fce1 ab3fce1d4a88e7c6a6b9d8be823fa9dedab97247 -m 'R26 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/R27-00343c1 00343c121a18ec4478cafad53237b97255b1955b -m 'R27 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/R27-ab3fce1 ab3fce1d4a88e7c6a6b9d8be823fa9dedab97247 -m 'R27 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/R28-3e55293 3e55293a443f04255cdecb86e58b7ab781ff2a67 -m 'R28 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/R28-ab3fce1 ab3fce1d4a88e7c6a6b9d8be823fa9dedab97247 -m 'R28 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/S1-8f017c3 8f017c3d62d2d82a4d581940a805acc952d12152 -m 'S1 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/S1-ab3fce1 ab3fce1d4a88e7c6a6b9d8be823fa9dedab97247 -m 'S1 的方法用过的 commit，不在 main 上（#85）'
git tag -a exp/S2-8f017c3 8f017c3d62d2d82a4d581940a805acc952d12152 -m 'S2 的方法用过的 commit，不在 main 上（#85）'
git push origin refs/tags/exp/B0-23d65aa refs/tags/exp/B0-4b2a395 refs/tags/exp/B0-563b738 refs/tags/exp/B0-6a75499 refs/tags/exp/B0-6c4e3f3 refs/tags/exp/B0-7224de2 refs/tags/exp/B0-cb293a8 refs/tags/exp/B0m-7224de2 refs/tags/exp/B0m-880b580 refs/tags/exp/B0m-9e85c61 refs/tags/exp/B0m-cb293a8 refs/tags/exp/R-0b64893 refs/tags/exp/R-76687c2 refs/tags/exp/R-e8a8e39 refs/tags/exp/R26-00343c1 refs/tags/exp/R26-ab3fce1 refs/tags/exp/R27-00343c1 refs/tags/exp/R27-ab3fce1 refs/tags/exp/R28-3e55293 refs/tags/exp/R28-ab3fce1 refs/tags/exp/S1-8f017c3 refs/tags/exp/S1-ab3fce1 refs/tags/exp/S2-8f017c3
