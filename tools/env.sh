# Sourced by tools/*.sh: loads the gitignored .env (template: .env.example) without executing it, keeping
# variables already set in the environment, then checks that the named variables are non-empty.
# Usage: . tools/env.sh VAR...
_hc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_hc_root/.env" ]; then
  while IFS= read -r _hc_line || [ -n "$_hc_line" ]; do
    _hc_line=${_hc_line%$'\r'}
    _hc_line=${_hc_line#"${_hc_line%%[![:space:]]*}"}
    case "$_hc_line" in '' | '#'*) continue ;; *=*) ;; *) continue ;; esac
    _hc_key=${_hc_line%%=*}
    _hc_key=${_hc_key#export }
    case "$_hc_key" in '' | *[!A-Za-z0-9_]*) continue ;; esac
    _hc_val=${_hc_line#*=}
    case "$_hc_val" in \"*\" | \'*\') _hc_val=${_hc_val:1:${#_hc_val}-2} ;; esac
    if [ -z "$(printenv "$_hc_key")" ]; then export "$_hc_key=$_hc_val"; fi
  done < "$_hc_root/.env"
fi
for _hc_var in "$@"; do
  if [ -z "$(printenv "$_hc_var")" ]; then
    echo "$_hc_var is not set: copy .env.example to .env and fill it in" >&2
    exit 1
  fi
done
unset _hc_root _hc_line _hc_key _hc_val _hc_var
