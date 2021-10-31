explain select count(*) from title t, movie_info mi, movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and mi.info_type_id > 16 and mi_idx.info_type_id = 100;
