#!/usr/bin/env bash
# Repair Vast.ai SSH key file permissions from the on-start script.
# This is intentionally standalone so it can be fetched with curl before
# cloning the repository or starting the training environment.

set -u

LOG_FILE="${VAST_SSH_REPAIR_LOG:-/tmp/vast_ssh_repair.log}"

repair_once() {
  mkdir -p /root/.ssh 2>/dev/null || true

  chown root:root /root 2>/dev/null || true
  chmod go-w /root 2>/dev/null || true

  chown root:root /root/.ssh 2>/dev/null || true
  chmod 700 /root/.ssh 2>/dev/null || true

  if [ -f /root/.ssh/authorized_keys ]; then
    chown root:root /root/.ssh/authorized_keys 2>/dev/null || true
    chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
  fi
}

log_state() {
  {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
    id
    stat -c "%U:%G %a %n" /root /root/.ssh /root/.ssh/authorized_keys 2>/dev/null || true
    command -v sshd >/dev/null 2>&1 && sshd -t 2>&1 || true
  } >>"${LOG_FILE}" 2>&1
}

repair_once
log_state

(
  # Vast may create or rewrite authorized_keys after the user on-start script
  # begins. Keep correcting it during the startup window so sshd sees safe
  # ownership and modes when the first login attempt arrives.
  for _ in $(seq 1 180); do
    repair_once
    sleep 1
  done
  log_state
) >>"${LOG_FILE}" 2>&1 &

echo "[vast-ssh-repair] started; log=${LOG_FILE}"
