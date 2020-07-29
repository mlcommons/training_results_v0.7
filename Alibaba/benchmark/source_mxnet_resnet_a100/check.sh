export LOGDEST="/data1/weiwei/scripts-v2/logA100" && echo "LOGDEST=${LOGDEST}"
export COMPLIANCESCRIPTS="/data1/weiwei/scripts-v2/logA100/mlperf_logging" && echo "COMPLIANCESCRIPTS=${COMPLIANCESCRIPTS}"
git clone --depth=1 "https://github.com/mlperf/logging" "${COMPLIANCESCRIPTS}" || true
cd ${COMPLIANCESCRIPTS}
git fetch origin || true
git reset --hard ${COMPLIANCE_VERSION}
git clean -f
find ${LOGDEST} -name "20*_*.log" > ${COMPLIANCESCRIPTS}/file.list
cat file.list | wc -l
for l in `cat file.list`
do python3.6 -m mlperf_logging.compliance_checker $l
done
