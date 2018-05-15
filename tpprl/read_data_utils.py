import bisect


def prune_sim_opts(sim_opts, followee_ids, start_time, end_time):
    """Trims off sim_opts by removing the sources whose src_id is not
    in the followee_ids set."""

    # Actually, we can strip the times of the other sources to reduce the size of
    # the dataset significantly further.

    other_sources = []
    for (kind, d) in sim_opts.other_sources:
        if kind == 'RealData' and d['src_id'] in followee_ids:
            d2 = d.copy()
            start_idx, end_idx = bisect.bisect(d['times'], start_time), bisect.bisect(d['times'], end_time)
            d2['times'] = d['times'][start_idx:end_idx]
            other_sources.append((kind, d2))

    edge_list = [(followee_id, follower_id) for (followee_id, follower_id) in sim_opts.edge_list
                 if followee_id in followee_ids or followee_id == sim_opts.src_id]

    sink_ids = sorted(set([follower_id for (_, follower_id) in edge_list]))

    return sim_opts.update({
        'other_sources': other_sources,
        'edge_list': edge_list,
        'sink_ids': sink_ids,
        'end_time': end_time,
    })


def prune_one_user_data(one_user_data):
    """This is to strip the raw twitter data to such a form that it only contains events corresponding
    to the main broadcasters actions."""

    assert 'followees' in one_user_data, "one_user_data should have the list of allowed followers within it."
    user_event_times = one_user_data['user_event_times']
    start_time, end_time = user_event_times[0], user_event_times[-1]

    # Removes the followers/sources which we do not need.
    sim_opts = prune_sim_opts(one_user_data['sim_opts'], one_user_data['followees'], start_time, end_time)

    new_user_data = one_user_data.copy()
    new_user_data['sim_opts'] = sim_opts
    return new_user_data



