import bisect
from redqueen.utils import def_s_vec


def prune_sim_opts_by_follower(sim_opts, follower_ids, followee_ids, start_time, end_time):
    """Trims off sim_opts by removing the sources whose src_id is not
    in the followee_ids set."""

    follower_ids = set(follower_ids)
    followee_ids = set([src for (src, dst) in sim_opts.edge_list
                        if dst in follower_ids]).intersection(followee_ids)

    other_sources = []
    seen_src = set()

    for (kind, d) in sim_opts.other_sources:
        if kind == 'RealData' and d['src_id'] in followee_ids and d['src_id'] not in seen_src:
            seen_src.add(d['src_id'])  # De-dup sources.
            d2 = d.copy()
            start_idx, end_idx = bisect.bisect(d['times'], start_time), bisect.bisect(d['times'], end_time)
            d2['times'] = d['times'][start_idx:end_idx]
            other_sources.append((kind, d2))

    edge_list = [(followee_id, follower_id) for (followee_id, follower_id) in sim_opts.edge_list
                 if follower_id in follower_ids and (
                     followee_id == sim_opts.src_id or
                     followee_id in followee_ids)
                 ]

    return sim_opts.update({
        'other_sources': other_sources,
        'edge_list': edge_list,
        'sink_ids': follower_ids,
        'end_time': end_time,
        's': def_s_vec(len(follower_ids))
    })


def prune_sim_opts_by_followee(sim_opts, followee_ids, start_time, end_time):
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
        's': def_s_vec(len(sink_ids))
    })


def prune_one_user_data(one_user_data):
    """This is to strip the raw twitter data to such a form that it only contains events corresponding
    to the main broadcasters actions."""

    assert 'followees' in one_user_data, "one_user_data should have the list of allowed followers within it."
    user_event_times = one_user_data['user_event_times']
    start_time, end_time = user_event_times[0], user_event_times[-1]

    # Removes the followers/sources which we do not need.
    sim_opts = prune_sim_opts_by_followee(
        sim_opts=one_user_data['sim_opts'],
        followee_ids=one_user_data['followees'],
        start_time=start_time,
        end_time=end_time
    )

    new_user_data = one_user_data.copy()
    new_user_data['sim_opts'] = sim_opts
    return new_user_data


def merge_lonely_sources(one_user_data, verbose=False):
    """For each wall, merges all broadcasters who only send messages to this
    wall into one for efficiency and to make the learning problem less
    complex."""
    sim_opts = one_user_data['sim_opts']

    new_other_sources = []
    new_edge_list = []
    all_shared_src_ids = {}

    src_id_to_b_dict = {broadcaster['src_id']: (_kind, broadcaster)
                        for _kind, broadcaster in sim_opts.other_sources}

    for sink_id in sim_opts.sink_ids:
        all_source_ids = set(src for src, dst in sim_opts.edge_list
                             if dst == sink_id and src != sim_opts.src_id)
        shared_source_ids = all_source_ids.intersection(
            set(src for src, dst in sim_opts.edge_list
                if dst != sink_id and
                src != sim_opts.src_id)
        )
        all_shared_src_ids[sink_id] = shared_source_ids
        lonely_source_ids = all_source_ids.difference(shared_source_ids)

        if len(lonely_source_ids) > 0:
            lonely_src_id = min(lonely_source_ids)
            all_times = sorted([t
                                for (_kind, broadcaster) in sim_opts.other_sources
                                if broadcaster['src_id'] in lonely_source_ids
                                for t in broadcaster['times']])
            kind, d = src_id_to_b_dict[lonely_src_id]
            d2 = d.copy()
            d2['times'] = all_times

            new_other_sources.append((kind, d2))
            new_edge_list.append((lonely_src_id, sink_id))
        else:
            # This wall has only shared sources
            pass

    seen_src_id = set()   # For another de-dup
    for sink_id, shared_src_ids in all_shared_src_ids.items():
        for src_id in shared_src_ids:
            if src_id not in seen_src_id:
                seen_src_id.add(src_id)
                new_other_sources.append(src_id_to_b_dict[src_id])
                new_edge_list.append((src_id, sink_id))

    ret_data = one_user_data.copy()
    ret_data['sim_opts'] = sim_opts.update({
        'other_sources': new_other_sources,
        'edge_list': new_edge_list,
    })
    return one_user_data


def merge_sinks(one_user_data):
    """Merges all sinks into one single wall."""
    merged_sinks_data = one_user_data.copy()
    sink_id = 999

    new_edge_list = [(one_user_data['user_id'], sink_id)]
    for (k, d) in one_user_data['sim_opts'].other_sources:
        new_edge_list.append((d['src_id'], sink_id))

    new_sim_opts = one_user_data['sim_opts'].update({
        'edge_list': new_edge_list,
        'sink_ids': [sink_id],
        's': 1.0
    })

    merged_sinks_data['sim_opts'] = new_sim_opts
    return merged_sinks_data
