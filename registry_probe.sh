#!/usr/bin/env bash
set -u

for registry_url in \
  https://docker.m.daocloud.io/v2/ \
  https://dockerproxy.net/v2/ \
  https://docker.1panel.live/v2/ \
  https://hub.rat.dev/v2/ \
  https://docker.chenby.cn/v2/ \
  https://docker.1ms.run/v2/ \
  https://mirror.baidubce.com/v2/
do
  result="$(curl -L -sS -o /dev/null -w '%{http_code} %{time_total}' --connect-timeout 5 --max-time 12 "${registry_url}" 2>/dev/null || true)"
  printf '%s %s\n' "${result}" "${registry_url}"
done
