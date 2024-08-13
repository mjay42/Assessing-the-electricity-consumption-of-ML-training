#!/usr/bin/env bash

service_name="ilo_power";

csv_path="$1/ilo_power.csv";

echo "timestamp,power_watt" > $csv_path;

while true; do

  power=`ipmitool dcmi power reading 5_sec | awk -F '.*Instantaneous power reading: +' '/Instantaneous power reading:\s*(.*)/ {print $2}' | awk '{print $1}'`
  timestamp_ms=$(date +%s)
  echo "$timestamp_ms,$power"  >> $csv_path;
  sleep 10

done
