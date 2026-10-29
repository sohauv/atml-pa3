import time
from common.modal_config import app, data_image, data_volume

prepare_image = data_image.add_local_python_source("common", "task1_finetuning")


@app.function(image=prepare_image, cpu=(4, 4), memory=(32768, 32768),
              timeout=1800, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/data": data_volume})
def prepare_execute(request):
    if request["ticket"]["deadline"] - time.time() < 120:
        raise TimeoutError("Preparation reservation expired")
    data_volume.reload()
    if request["job"] == "verify_cache2":
        from common.cache_profile import verify_feature_cache
        from pathlib import Path
        from common.io import write_json
        result = verify_feature_cache(request["features"], max_seconds=request["ticket"]["deadline"] - time.time(),
                                      verified_index=request.get("verified_index"))
        write_json(Path("/data/task2/cpu-profiles") / (request["run_id"] + ".json"), result)
    elif request["job"] == "profile_cache2":
        from pathlib import Path
        from common.cache_profile import profile_feature
        from common.io import write_json
        samples = Path(request["features"]).glob("*.pt")
        sample = next(samples)
        result = profile_feature(sample, comparison_path=next(samples, None))
        write_json(Path("/data/task2/cpu-profiles") / (request["run_id"] + ".json"), result)
    else:
        from task1_finetuning.prepare_data import prepare_data
        result = prepare_data()
    data_volume.commit()
    return result
