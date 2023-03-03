import copy

from uptrain.core.lib.helper_funcs import extract_data_points_from_batch
from uptrain.core.classes.statistics import AbstractStatistic
from uptrain.core.classes.distances import DistanceResolver
from uptrain.core.classes.measurables import MeasurableResolver
from uptrain.constants import Statistic
from uptrain.core.lib.duckdb import fetch_values_for_id, upsert_id_values

import duckdb
import numpy as np
import os


class Distance(AbstractStatistic):
    dashboard_name = "distance"
    anomaly_type = Statistic.DISTANCE

    def __init__(self, fw, check):
        self.allowed_model_values = [x['allowed_values'] for x in check['model_args']]
        self.num_model_options = sum([len(x) > 1 for x in self.allowed_model_values])
        self.children = []
        self.unique_id = fw.check_class_id
        fw.check_class_id += 1

        if self.num_model_options > 0:
            for m in self.allowed_model_values[0]:
                check_copy = copy.deepcopy(check)
                check_copy['model_args'][0]['allowed_values'] = [m]
                check_copy['model_args'].append(copy.deepcopy(check_copy['model_args'][0]))
                del check_copy['model_args'][0]
                self.children.append(Distance(fw, check_copy))
        else:
            self.log_handler = fw.log_handler
            self.measurable = MeasurableResolver(check["measurable_args"]).resolve(fw)
            self.aggregate_measurable = MeasurableResolver(check["aggregate_args"]).resolve(
                fw
            )
            self.count_measurable = MeasurableResolver(check["count_args"]).resolve(
                fw
            )
            self.feature_measurables = [
                MeasurableResolver(x).resolve(fw) for x in check["feature_args"]
            ]
            self.model_measurables = [
                MeasurableResolver(x).resolve(fw) for x in check["model_args"]
            ]
            self.model_names = [x.col_name() for x in self.model_measurables]
            self.feature_names = [x.col_name() for x in self.feature_measurables]
            self.reference = check["reference"]
            self.distance_types = check["distance_types"]
            self.dist_classes = [DistanceResolver().resolve(x) for x in self.distance_types]

            db_dir = os.path.join(fw.fold_name, 'dbs', 'distance')
            os.makedirs(db_dir, exist_ok=True)            
            self.conn = duckdb.connect(os.path.join(db_dir, f"ref_embs.db"))
            self.ref_table = f'ref_embs_id_{self.unique_id}'
            self.conn.execute(f"CREATE OR REPLACE TABLE {self.ref_table} (id LONG PRIMARY KEY, value FLOAT[])")
            # self.conn.execute("CREATE OR REPLACE TABLE id_counts (id LONG, value FLOAT)")

    def check(self, inputs, outputs=None, gts=None, extra_args={}):
        if len(self.children) > 0:
            [x.check(inputs, outputs, gts, extra_args) for x in self.children]
        else:
            vals_all = self.measurable.compute_and_log(
                inputs, outputs, gts=gts, extra=extra_args
            )
            aggregate_ids_all = self.aggregate_measurable.compute_and_log(
                inputs, outputs, gts=gts, extra=extra_args
            )
            counts = self.count_measurable.compute_and_log(inputs, outputs, gts=gts, extra=extra_args)
            all_models = [x.compute_and_log(
                inputs, outputs, gts=gts, extra=extra_args
            ) for x in self.model_measurables]
            all_features = [x.compute_and_log(
                inputs, outputs, gts=gts, extra=extra_args
            ) for x in self.feature_measurables]

            idxs = []
            for idx in range(len(aggregate_ids_all)):
                is_model_invalid = sum([all_models[jdx][idx] not in self.allowed_model_values[jdx] for jdx in range(len(self.allowed_model_values))])
                if not is_model_invalid:
                    idxs.append(idx)            
                
            aggregate_ids = aggregate_ids_all[idxs]
            vals = vals_all[idxs, :]
         
            ref_embs_tbl = fetch_values_for_id(self.conn, self.ref_table, aggregate_ids)
            ref_embs_remote = dict(zip(ref_embs_tbl['id'], ref_embs_tbl['value']))

            ref_embs = {}
            ref_embs_prev = {}
            ref_vals = None
            for i, key in enumerate(aggregate_ids):
                if key not in ref_embs:
                    ref_embs.update({key: ref_embs_remote.get(key, vals[i])})
                if self.reference=='running_diff':
                    if key in ref_embs_prev:
                        ref_embs.update({key: ref_embs_prev[key]})
                    ref_embs_prev.update({key: vals[i]})
                ref_vals = ref_embs[key] if ref_vals is None else np.vstack([ref_vals, ref_embs[key]])
            ref_vals = np.array(ref_vals)

            # ref_vals = np.array([ref_embs[key] for key in aggregate_ids])
            if len(vals) > 0:
                try:
                    if len(ref_vals.shape) == 1:
                        ref_vals = np.expand_dims(ref_vals, 0)
                except:
                    import pdb; pdb.set_trace()
                distances = dict(zip(self.distance_types, 
                                     [x.compute_distance(vals, ref_vals) for x in self.dist_classes]))

            for idx in range(len(aggregate_ids)):
                # this_val = extract_data_points_from_batch(vals, [idx])

                # key = aggregate_ids[idx]
                # ref_emb = ref_embs[key]
                # this_distances = dict(
                #     zip(
                #         self.distance_types,
                #         [
                #             x.compute_distance(
                #                 this_val, [ref_emb]
                #             )
                #             for x in self.dist_classes
                #         ],
                #     )
                # )
                models = dict(zip(['model_' + x for x in self.model_names], [all_models[jdx][idx] for jdx in range(len(self.model_names))]))
                features = dict(zip(['feature_' + x for x in self.feature_names], [all_features[jdx][idx] for jdx in range(len(self.feature_names))]))
                for distance_type in self.distance_types:
                    plot_name = (
                        distance_type
                        + "_"
                        + str(self.reference)
                    )
                    self.log_handler.add_scalars(
                        self.dashboard_name + "_" + plot_name,
                        {'y_' + distance_type: distances[distance_type][idx]},
                        counts[idx],
                        self.dashboard_name,
                        features = features,
                        models = models,
                        file_name = str(aggregate_ids[idx])
                    )

            if self.reference == "running_diff":
                ref_embs = dict(zip(aggregate_ids, vals))
            if ref_embs:
                keys, values = zip(*ref_embs.items())
                upsert_id_values(self.conn, self.ref_table, np.array(keys), np.array(values))
                

