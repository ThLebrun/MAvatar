def perf_SDV(data1, data2):
    from sdv.metadata import SingleTableMetadata
    from sdmetrics.reports.single_table import QualityReport

    metadata_object = SingleTableMetadata()

    metadata_object.detect_from_dataframe(data=data1)
    metadata = metadata_object.to_dict()
    report = QualityReport()
    report.generate(data1, data2, metadata)

    perfs = report.get_properties()
    return {perfs.Property[i]: perfs.Score[i] for i in range(len(perfs))}


def SDV(pb, algo, seed):
    dict_results = {}
    data, _, syn = exp_plan_loader(pb, algo, seed)
    data.reset_index(inplace=True, drop=True)
    syn.reset_index(inplace=True, drop=True)
    perf = perf_SDV(data, syn)
    dict_results["Column Shapes"] = perf["Column Shapes"]
    dict_results["Column Pair Trends"] = perf["Column Pair Trends"]
    return dict_results
