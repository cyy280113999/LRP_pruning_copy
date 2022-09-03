def analyze_log(args):
    # args is a argparse.Namespace object.
    assert os.path.isfile(args.logfile), 'Error! no log file to analyze at "{}"'.format(args.logfile)
    logdir = os.path.dirname(args.logfile)  # draw figure next to analzed log, ie within this folder

    # some helper fxns
    def concat_helper(lists):
        return lists[0] + lists[1]  # helper fxn due to the inavailability of unpacking inlist comprehensions

    def color_per_criterion(crit):
        return {'lrp': 'red', 'weight': 'black', 'grad': 'green', 'taylor': 'blue'}[crit]

    def correlation(data1, data2, measure='spearman'):
        return {'spearman': spearmanr, 'kendalltau': kendalltau}[measure](data1, data2)[
            0]  # index 0: only collect correlation, not p-value

    def add_to_dict(result_dict, keylist, value):
        # dynamically expands a dictionary
        if len(keylist) == 1:
            if keylist[0] in result_dict:
                result_dict[keylist[0]] += [value]
            else:
                result_dict[keylist[0]] = [value]
        else:
            if not keylist[0] in result_dict:
                result_dict[keylist[0]] = {}
            add_to_dict(result_dict[keylist[0]], keylist[1::], value)

    # read and parse the log.
    # sample log line:
    # dataset:circle-criterion:weight-n:2-s:0-scenario:train-stage:pre 100.0
    with open(args.logfile, 'rt') as f:
        data = f.read().split('\n')
        data = [concat_helper([[c.split(':')[-1] for c in w.split('-')] for w in l.split()]) for l in data if
                len(l) > 0]

    # subscriptable array with field indices as below
    dset, crit, n, seed, scenario, stage, value = range(7)  # field names as indices
    dset_t, crit_t, n_t, seed_t, scenario_t, stage_t, value_t = str, str, float, float, str, str, float  # "natural" data types per field. final float assumes "accuracy" case.
    data = np.array(data)

    # (re)normalize sample count to "per class"
    data[:, n] = data[:, n].astype(n_t) / (2 + 2 * (data[:, dset] == 'mult'))

    if args.ranklog:
        # tables to produce
        #
        # over n in reference_sample_counts:
        #     - rank corellation
        #     method vs method (except for identical seed, except for weight). one table, since for computing the rank, one has to consider all neurons
        #
        #    for k in set_sizes_of_k
        #        - set intersection
        #        method vs method, for different set sizes of the first k (= least important  k, also last k = most important k) neurons/filters

        # filter out irrelevant stuff
        data = data[data[:, scenario] == 'rankselection']
        seeds = np.unique(data[:, seed])  # NOTE set to or lower [:10] for debugging
        corellation_measures = ['spearman', 'kendalltau']
        set_sizes = [125, 250, 500, 1000]  # set sizes for intersection computation

        print('Computing Rank Corellation and Set Intersection Scores')
        for dset_name in tqdm.tqdm(valid_datasets, desc="datasets",
                                   leave=False):  # approach: progressively filter out data to mimize search times a iteration depth increases
            current_data = data[data[:, dset] == dset_name]
            assert current_data.shape[0] > 0, "Error! current_data empty after dset filtering"
            for n_samples in tqdm.tqdm(np.unique(data[:, n])[np.argsort(np.unique(data[:, n]).astype(n_t))],
                                       desc="sample counts", leave=False):
                current_data_n = current_data[current_data[:, n] == n_samples]
                assert current_data_n.shape[0] > 0, "Error! current_data_n empty after sample filtering"
                # gather data and compute three kinds of rank corellation and set Itersection (case four can be found further below)
                # 1) c1==c2 and s1!=s2: consistency comparison across different seeds  (else they are identical).
                #   一致性

                # 2) c1==c2 and s1==s2 or c1!=c2: cross criteria similarity comparison across all seeds
                #   相似性

                # 3) s1 == s2: cross criteria comparison across same seed
                case_1_results = {}  # {c1:{c2:{correlation_or_intersection_measure:[list over all the seeds with s1 != s2]}}}
                case_2_results = {}  # {c1:{c2:{correlation_or_intersection_measure:[list over all the seeds]}}}
                case_3_results = {}  # {c1:{c2:{correlation_or_intersection_measure:[list over all the seeds with s1 == s2]}}}

                for c1, c2 in tqdm.tqdm(list(itertools.product(valid_criteria, valid_criteria)),
                                        desc='criteria combinations', leave=False):
                    c1_data = current_data_n[(current_data_n[:, crit] == c1)]
                    c2_data = current_data_n[(current_data_n[:, crit] == c2)]
                    assert c1_data.shape[0] > 0, "Error! c1_data empty after criterion filtering"
                    assert c2_data.shape[0] > 0, "Error! c2_data empty after criterion filtering"

                    for s1, s2 in tqdm.tqdm(list(itertools.product(seeds, seeds)), desc="random seed combinations",
                                            leave=False):
                        data1 = json.loads(c1_data[c1_data[:, seed] == s1][0, value])
                        data2 = json.loads(c2_data[c2_data[:, seed] == s2][0, value])
                        scorr = correlation(data1, data2, 'spearman')
                        kcorr = correlation(data1, data2, 'kendalltau')

                        tmp_set_intersections = {}  # {first-<k>/last-<k>:[list over all seeds]}
                        for k in set_sizes:
                            firstk_intersection_coverage = len(set(data1[:k]).intersection(data2[:k])) / k
                            lastk_intersection_coverage = len(set(data1[-k:]).intersection(data2[-k:])) / k
                            tmp_set_intersections['first-{}'.format(k)] = firstk_intersection_coverage
                            tmp_set_intersections['last-{}'.format(k)] = lastk_intersection_coverage

                        if c1 == c2 and s1 != s2:  # case 1
                            add_to_dict(case_1_results, [c1, c2, 'spearman'], scorr)
                            add_to_dict(case_1_results, [c1, c2, 'kendalltau'], kcorr)
                            for k in tmp_set_intersections.keys():
                                add_to_dict(case_1_results, [c1, c2, k], tmp_set_intersections[k])

                        if c1 == c2 and s1 == s2 or c1 != c2:  # case 2 & case 3
                            add_to_dict(case_2_results, [c1, c2, 'spearman'], scorr)
                            add_to_dict(case_2_results, [c1, c2, 'kendalltau'], kcorr)
                            for k in tmp_set_intersections.keys():
                                add_to_dict(case_2_results, [c1, c2, k], tmp_set_intersections[k])

                        if s1 == s2:  # case 3 only.
                            add_to_dict(case_3_results, [c1, c2, 'spearman'], scorr)
                            add_to_dict(case_3_results, [c1, c2, 'kendalltau'], kcorr)
                            for k in tmp_set_intersections.keys():
                                add_to_dict(case_3_results, [c1, c2, k], tmp_set_intersections[k])

                # whole tables per measure.
                # for each combination of dset and num_samples
                # write out case_results here.
                with open('{}/rank_and_set-{}.txt'.format(logdir, dset_name), 'at') as f:
                    ##
                    ## CASE 1 RESULTS
                    ##
                    header_template = f'# {dset_name} n={n_samples} case 1: self-compare criteria across random seeds -> check pruning consistency'.upper()
                    header_support = '#' * len(header_template)
                    header = '\n'.join([header_support, header_template, header_support])

                    # assumption: all pairs of stuff have been computed on the same things
                    criteria = list(case_1_results.keys())
                    all_measures = list(list(list(case_1_results.values())[0].values())[0].keys())

                    f.write(header)
                    f.write('\n' * 3)

                    for m in all_measures:
                        t = PrettyTable()
                        t.field_names = [m] + criteria
                        for i, c in enumerate(criteria):
                            val = np.mean(case_1_results[c][c][m])
                            std = np.std(case_1_results[c][c][m])
                            t.add_row([c] + i * [''] + ['{:.3f}+-{:.3f}'.format(val, std)] + [''] * (
                                    len(criteria) - 1 - i))

                        f.write(str(t))
                        f.write('\n' * 2)

                    ##
                    ## CASE 2
                    ##
                    header_template = f'# {dset_name} n={n_samples} case 2: cross-compare criteria across all random seeds -> check relationship between criteria'.upper()
                    header_support = '#' * len(header_template)
                    header = '\n'.join([header_support, header_template, header_support])

                    f.write(header)
                    f.write('\n' * 3)

                    for m in all_measures:
                        t = PrettyTable()
                        t.field_names = [m] + criteria
                        for cr in criteria:
                            row = [cr]
                            for cc in criteria:
                                val = np.mean(case_2_results[cr][cc][m])
                                std = np.std(case_2_results[cr][cc][m])
                                row += ['{:.3f}+-{:.3f}'.format(val, std)]
                            t.add_row(row)

                        f.write(str(t))
                        f.write('\n' * 2)

                    ##
                    ## CASE 3
                    ##
                    header_template = '# {} n={} case 3 cross-compare criteria, same random seeds only! -> check relationship between criteria wrt same data source'.format(
                        dset_name, n_samples).upper()
                    header_support = '#' * len(header_template)
                    header = '\n'.join([header_support, header_template, header_support])

                    f.write(header)
                    f.write('\n' * 3)

                    for m in all_measures:
                        t = PrettyTable()
                        t.field_names = [m] + criteria
                        for cr in criteria:
                            row = [cr]
                            for cc in criteria:
                                val = np.mean(case_3_results[cr][cc][m])
                                std = np.std(case_3_results[cr][cc][m])
                                row += ['{:.3f}+-{:.3f}'.format(val, std)]
                            t.add_row(row)

                        f.write(str(t))
                        f.write('\n' * 2)

            #
            # Compute cases 4 and 5
            # in addition to above result sets
            #
            for c in tqdm.tqdm(valid_criteria, desc="global criteria consistency", leave=False):
                # 4) and 5) comparison of neuron rank order for one method, across sample sizes
                case_4_results = {}  # {n1:{n2:{correlation_or_intersection_measure:[list over all the seeds with s1 != s2 or s1 == 2]}}}
                case_5_results = {}  # {n1:{n2:{correlation_or_intersection_measure:[list over all the seeds with s1 == 2]}}}
                current_data_c = current_data[current_data[:, crit] == c]
                assert current_data_c.shape[0] > 0, "Error! current_data_c empty after sample filtering"

                for n1, n2 in tqdm.tqdm(list(itertools.product(np.unique(data[:, n]), np.unique(data[:, n]))),
                                        desc="sample set size combinations", leave=False):
                    n1_data = current_data_c[current_data_c[:, n] == n1]
                    n2_data = current_data_c[current_data_c[:, n] == n2]
                    assert n1_data.shape[0] > 0, "Error! n1_data empty after criterion filtering"
                    assert n2_data.shape[0] > 0, "Error! n2_data empty after criterion filtering"

                    for s1, s2 in tqdm.tqdm(list(itertools.product(seeds, seeds)), desc="random seed combinations",
                                            leave=False):
                        data1 = json.loads(n1_data[n1_data[:, seed] == s1][0, value])
                        data2 = json.loads(n2_data[n2_data[:, seed] == s2][0, value])
                        scorr = correlation(data1, data2, 'spearman')
                        kcorr = correlation(data1, data2, 'kendalltau')

                        tmp_set_intersections = {}  # {first-<k>/last-<k>:[list over all seeds]}
                        for k in set_sizes:
                            firstk_intersection_coverage = len(set(data1[:k]).intersection(data2[:k])) / k
                            lastk_intersection_coverage = len(set(data1[-k:]).intersection(data2[-k:])) / k
                            tmp_set_intersections['first-{}'.format(k)] = firstk_intersection_coverage
                            tmp_set_intersections['last-{}'.format(k)] = lastk_intersection_coverage

                        add_to_dict(case_4_results, [n1, n2, 'spearman'], scorr)
                        add_to_dict(case_4_results, [n1, n2, 'kendalltau'], kcorr)
                        for k in tmp_set_intersections.keys():
                            add_to_dict(case_4_results, [n1, n2, k], tmp_set_intersections[k])

                        if s1 == s2:
                            add_to_dict(case_5_results, [n1, n2, 'spearman'], scorr)
                            add_to_dict(case_5_results, [n1, n2, 'kendalltau'], kcorr)
                            for k in tmp_set_intersections.keys():
                                add_to_dict(case_5_results, [n1, n2, k], tmp_set_intersections[k])

                # write out results for  cases 4 and 5
                # whole tables per criterion.
                # for each combination of num_samples, over the seeds
                # write out case_results here.
                with open('{}/rank_and_set-{}.txt'.format(logdir, dset_name), 'at') as f:
                    ##
                    ## CASE 4 RESULTS
                    ##
                    header_template = f'# {dset_name} and {c} case 4: self-compare criteria across sample sizes (and random seeds) -> check pruning consistency'.upper()
                    header_support = '#' * len(header_template)
                    header = '\n'.join([header_support, header_template, header_support])

                    # assumption: all pairs of stuff have been computed on the same things
                    criterion = c
                    all_n_samples = np.array(list(case_4_results.keys()))
                    all_n_samples = list(
                        all_n_samples[np.argsort(all_n_samples.astype(np.float32))])  # order ascendingly
                    all_measures = list(list(list(case_4_results.values())[0].values())[0].keys())

                    f.write(header)
                    f.write('\n' * 3)

                    for m in all_measures:
                        t = PrettyTable()
                        t.field_names = ['{}:{}'.format(criterion, m)] + all_n_samples
                        for nr in all_n_samples:
                            row = [nr]
                            for nc in all_n_samples:
                                val = np.mean(case_4_results[nr][nc][m])
                                std = np.std(case_4_results[nr][nc][m])
                                row += ['{:.3f}+-{:.3f}'.format(val, std)]
                            t.add_row(row)

                        f.write(str(t))
                        f.write('\n' * 2)

                    ##
                    ## CASE 5 RESULTS
                    ##
                    header_template = '# {} and {} case 5: self-compare criteria across sample sizes (same seed only) -> check pruning consistency'.format(
                        dset_name, c).upper()
                    header_support = '#' * len(header_template)
                    header = '\n'.join([header_support, header_template, header_support])

                    f.write(header)
                    f.write('\n' * 3)

                    for m in all_measures:
                        t = PrettyTable()
                        t.field_names = ['{}:{}'.format(criterion, m)] + all_n_samples
                        for nr in all_n_samples:
                            row = [nr]
                            for nc in all_n_samples:
                                val = np.mean(case_5_results[nr][nc][m])
                                std = np.std(case_5_results[nr][nc][m])
                                row += ['{:.3f}+-{:.3f}'.format(val, std)]
                            t.add_row(row)

                        f.write(str(t))
                        f.write('\n' * 2)




    else:
        # analyze accuracy post-pruning wrt n and criterion here

        # now draw some line plots
        for dset_name in np.unique(data[:, dset]):
            for scenario_name in ['train', 'test']:
                fig = plt.figure(figsize=(3.5, 3.5))
                plt.subplots_adjust(left=0.19, right=0.99, top=0.93, bottom=0.13)
                plt.title('{}, on {} data'.format(dset_name, scenario_name))
                plt.xlabel('samples used to compute criteria')
                plt.ylabel('performance after pruning in %')
                plt.xscale('log')  # show x-axis in log scale, as suggested by reviewer
                plt.gca().tick_params(which='minor', length=0)  # hide minor log scale ticks

                current_data = data[(data[:, dset] == dset_name) * (data[:,
                                                                    scenario] == scenario_name)]  # get the currently relevant data for "this" line plot
                x = current_data[:, n].astype(n_t)
                x_min = x.min()
                x_max = x.max()

                # draw baseline (original model performance, as average with standard deviation (required for test setting))
                # y_baseline = current_data[current_data[:,stage] == 'pre'][:,acc].astype(acc_t)
                data_baseline = current_data[current_data[:, stage] == 'pre']
                x_baseline = data_baseline[:, n]

                # compute average values for y per x.
                y_baseline_avg = np.array(
                    [np.mean(data_baseline[data_baseline[:, n] == xi, value].astype(value_t)) for xi in
                     np.unique(x_baseline)])
                y_baseline_std = np.array(
                    [np.std(data_baseline[data_baseline[:, n] == xi, value].astype(value_t)) for xi in
                     np.unique(x_baseline)])
                x_baseline = np.unique(x_baseline).astype(n_t)

                # sort wrt ascending x
                ii = np.argsort(x_baseline)
                x_baseline = x_baseline[ii]
                y_baseline_avg = y_baseline_avg[ii]
                y_baseline_std = y_baseline_std[ii]

                plt.fill_between(x_baseline, y_baseline_avg - y_baseline_std,
                                 np.minimum(y_baseline_avg + y_baseline_std, 100), color='black', alpha=0.2)
                plt.plot(x_baseline, y_baseline_avg, '--', color='black', label='no pruning')

                # print out some stats for the table/figure description
                for i in range(x_baseline.size):
                    print('dataset={}, stage=pre, no pruning, n={} : {} acc = {:.2f}'.format(dset_name, x[i],
                                                                                             scenario_name,
                                                                                             y_baseline_avg[i]))
                print()

                # draw achual model performance after pruning, m'lady *heavy breathing*
                for crit_name in np.unique(current_data[:, crit]):
                    tmp = current_data[(current_data[:, stage] == 'post') * (current_data[:, crit] == crit_name)]
                    x = tmp[:, n]

                    # compute average values for y per x.
                    y_avg = np.array([np.mean(tmp[tmp[:, n] == xi, value].astype(value_t)) for xi in np.unique(x)])
                    y_std = np.array([np.std(tmp[tmp[:, n] == xi, value].astype(value_t)) for xi in np.unique(x)])
                    x = np.unique(x).astype(n_t)

                    # sort wrt ascending x
                    ii = np.argsort(x)
                    x = x[ii]
                    y_avg = y_avg[ii]
                    y_std = y_std[ii]

                    # plot the lines
                    color = color_per_criterion(crit_name)
                    plt.fill_between(x, y_avg - y_std, np.minimum(y_avg + y_std, 100), color=color, alpha=0.2)
                    plt.plot(x, y_avg, color=color, label=crit_name)
                    # plt.xticks(x,[int(i) if i in [10,50,100,200] else '' for i in x], ha='right')
                    plt.xticks(x, [int(i) if i in [1, 2, 5, 10, 20, 50, 100, 200] else '' for i in x],
                               ha='right')  # more tick marks for log scale
                    plt.gca().xaxis.grid(True, 'major')
                    plt.legend(loc='lower right')

                    # print out some stats for the table/figure description
                    for i in range(x.size):
                        print('dataset={}, stage=post, crit={}, n={} : {} acc = {:.2f}'.format(dset_name, crit_name,
                                                                                               x[i], scenario_name,
                                                                                               y_avg[i]))
                    print()

                plt.xlim([x_min, x_max])
                # save figure

                if args.rendermode in ['svg']:
                    figname = '{}/{}-{}.svg'.format(logdir, dset_name, scenario_name)
                    print('Saving result figure to {}'.format(figname))
                    plt.savefig(figname)

                if args.rendermode in ['show']:
                    plt.show()

                plt.close()
