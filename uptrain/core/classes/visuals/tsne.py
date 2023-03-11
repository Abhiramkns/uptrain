import copy
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import numpy as np

from uptrain.core.classes.visuals import AbstractVisual
from uptrain.constants import Visual, Statistic
from uptrain.core.classes.measurables import MeasurableResolver
from uptrain.core.lib.helper_funcs import read_json, cluster_and_plot_data

class Tsne(AbstractVisual):
    visual_type = Visual.TSNE
    dashboard_name = "tsne_and_clusters"

    def base_init(self, fw, check):
        self.framework = fw
        self.label_measurable = MeasurableResolver(check.get("label_args", None)).resolve(
            fw
        )
        self.hover_measurables = [
            MeasurableResolver(x).resolve(fw) for x in check.get("hover_args", [])
        ]
        self.hover_names = [x.col_name() for x in self.hover_measurables]

        self.count_checkpoints = check.get("count_checkpoints", ["all"])
        self.dim = check.get("dim", '2D')
        self.perplexity = check.get("perplexity",30.0)
        self.early_exaggeration = check.get("early_exaggeration",12.0)
        self.learning_rate = check.get("learning_rate","auto")
        self.n_iter = check.get("n_iter",1000)
        self.n_iter_without_progress = check.get("n_iter_without_progress",300)
        self.min_grad_norm = check.get("min_grad_norm",1e-7)
        self.metric = check.get("metric",'euclidean')
        self.metric_params = check.get("metric_params",None)
        self.init = check.get("init","pca")
        self.verbose = check.get("verbose",0)
        self.random_state = check.get("random_state",None)
        self.method = check.get("method",'barnes_hut')
        self.angle = check.get("angle",0.5)
        self.n_jobs = check.get("n_jobs",None)
        self.min_samples = check.get("min_samples", 5)
        self.eps = check.get("eps", 2.0)
        self.total_count = 0
        self.prev_calc_at = 0
        self.update_freq = check.get('update_freq', 10000)
        self.vals = []
        self.labels = []
        self.hover_texts = []
        self.do_clustering = check.get("do_clustering", False)
        self.feature_dictn = {}

        self.initial_dataset = check.get('initial_dataset', None)
        if self.initial_dataset is not None:
            data = read_json(self.initial_dataset)
            self.vals.extend(
                [self.measurable.extract_val_from_training_data(x) for x in data]
            )
            if self.label_measurable is not None:
                self.labels.extend(
                    [self.label_measurable.extract_val_from_training_data(x) for x in data]
                )
            if len(self.hover_measurables):
                raw_data = np.abs([x['bert_embs_downsampled'] for x in data])
                self.max_along_axis = np.max(raw_data, axis=0)
                norm_data = raw_data/self.max_along_axis

                _, _, _, points_density, _ = cluster_and_plot_data(norm_data,20)

                offset = 0
                for data_point in data:
                    this_hovers = [x.extract_val_from_training_data(data_point) for x in self.hover_measurables]
                    self.hover_texts.append(dict(zip(self.hover_names, this_hovers)))
                    self.hover_texts[offset].update({"Training Density": int(points_density[offset])})
                    self.hover_texts[offset].update({"idx": offset})
                    offset += 1
            if len(self.feature_measurables):
                all_features = [list(np.reshape([feature_measurable.extract_val_from_training_data(x) for x in data], -1)) for feature_measurable in self.feature_measurables]
                this_dict = dict(zip(['feature_' + x for x in self.feature_names], all_features))
                for key in this_dict.keys():
                    if key in self.feature_dictn:
                        self.feature_dictn[key].extend(this_dict[key])
                    else:
                        self.feature_dictn.update({key: list(this_dict[key])})


    def base_check(self, inputs, outputs, gts=None, extra_args={}):
        if self.measurable is not None:
            vals = self.measurable.compute_and_log(
                inputs, outputs, gts=gts, extra=extra_args
            )
            self.vals.extend(vals)
        if self.label_measurable is not None:
            labels = self.label_measurable.compute_and_log(
                inputs, outputs, gts=gts, extra=extra_args
            )
            self.labels.extend(labels)

        if len(self.feature_measurables):
            all_features = [x.compute_and_log(
                inputs, outputs, gts=gts, extra=extra_args
            ) for x in self.feature_measurables]
            this_dict = dict(zip(['feature_' + x for x in self.feature_names], all_features))
            for key in this_dict.keys():
                if key in self.feature_dictn:
                    self.feature_dictn[key].extend(this_dict[key])
                else:
                    self.feature_dictn.update({key: list(this_dict[key])})

        if len(self.hover_measurables):
            hover_values = [x.compute_and_log(
                inputs, outputs, gts=gts, extra=extra_args
            ) for x in self.hover_measurables]
            hover_table = np.array(hover_values).transpose()
            self.hover_texts.extend([dict(zip(self.hover_names, values)) for values in hover_table])

        self.total_count += len(extra_args['id'])
        if not ((self.total_count - self.prev_calc_at) >= self.update_freq):
            return

        self.prev_calc_at = self.total_count
        models = dict(zip(['model_' + x for x in self.model_names], 
            [self.allowed_model_values[jdx][0] for jdx in range(len(self.model_names))]))
        for count in self.count_checkpoints:
            emb_list, label_list, hover_texts = self.get_high_dim_data(count)
            emb_list = np.array(emb_list)
            label_list = np.array(label_list)

            if emb_list.shape[0] > 10:
                clusters = []
                tsne_list, clusters = self.get_tsne_and_labels(
                    emb_list,
                    self.dim,
                    self.perplexity,
                    self.early_exaggeration,
                    self.learning_rate,
                    self.n_iter,
                    self.n_iter_without_progress,
                    self.min_grad_norm,
                    self.metric,
                    self.init,
                    self.verbose,
                    self.random_state,
                    self.method,
                    self.angle,
                    self.n_jobs,
                    self.eps,
                    self.min_samples,
                    label_list=label_list
                )
                this_data = {"umap": tsne_list, "clusters": clusters, "hover_texts": hover_texts}
                self.log_handler.add_histogram(
                    "tsne_and_clusters",
                    this_data,
                    self.dashboard_name,
                    models = models,
                    features = self.feature_dictn,
                    file_name = str(count) + "_" + '_'.join(list(models.values()))
                )

    def get_high_dim_data(self, count):
        if self.measurable is None:
            distribution_anomaly = list(
                filter(
                    lambda x: x.statistic_type == Statistic.DISTRIBUTION_STATS,
                    self.framework.check_manager.statistics_to_check,
                )
            )[0]
            data_dict = distribution_anomaly.get_feats_for_clustering(count, self.allowed_model_values)
            chosen_label_key = None
            chosen_hover_key = None
            if len(data_dict):
                temp_val = list(data_dict.values())[0]
                if len(temp_val):
                    temp_keys = list(temp_val.keys())
                    temp_keys = list(filter(lambda x: "visual_label_", temp_keys))
                    if len(temp_keys) > 1:
                        print("Have multiple labels - " + str(temp_keys) + " .Using " + temp_keys[0] + " for labeling TSNEs." )
                    if len(temp_keys) > 0:
                        chosen_label_key = temp_keys[0]
                if len(temp_val):
                    temp_keys = list(temp_val.keys())
                    temp_keys = list(filter(lambda x: "visual_hover_text_", temp_keys))
                    if len(temp_keys) > 1:
                        print("Have multiple hover texts - " + str(temp_keys) + " .Using " + temp_keys[0] + " for hovering TSNEs." )
                    if len(temp_keys) > 0:
                        chosen_hover_key = temp_keys[0]
            vals = np.array([data_dict[x]['val'] for x in data_dict])
            if vals.shape[0] > 0:
                vals = np.squeeze(vals, axis=1)
            labels = []
            if chosen_label_key is not None:
                labels = [data_dict[x][chosen_label_key] for x in data_dict]

            hover_texts = []
            if chosen_hover_key is not None:
                hover_texts = [{chosen_hover_key: data_dict[x][chosen_hover_key]} for x in data_dict]
            return vals, labels, hover_texts
        else:
            return self.vals, self.labels, self.hover_texts

    def get_tsne_and_labels(
        self,
        emb_list,
        dim,
        perplexity,
        early_exaggeration,
        learning_rate,
        n_iter,
        n_iter_without_progress,
        min_grad_norm,
        metric,
        init,
        verbose,
        random_state,
        method,
        angle,
        n_jobs,
        eps,
        min_samples,
        label_list=None
    ):
        if dim == "2D":
            n_components = 2
        else:
            n_components = 3

        embeddings = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            n_iter=n_iter,
            n_iter_without_progress=n_iter_without_progress,
            min_grad_norm=min_grad_norm,
            metric=metric,
            init=init,
            verbose=verbose,
            random_state=random_state,
            method=method,
            angle=angle,
            n_jobs=n_jobs
        ).fit_transform(emb_list)

        # Do DBSCAN clustering
        if self.do_clustering:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
            labels = clustering.labels_
        else:
            labels = label_list
        labels = np.squeeze(np.array(labels))
        return embeddings, labels
