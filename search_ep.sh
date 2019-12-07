# exec_file
exec_file=$1

for config_file in {"ep001","ep002","ep003","ep004","ep005","ep006","ep007","ep008","ep009"}; do
  echo "${exec_file}"
  bash "${exec_file}" ${config_file}
done
