UPDATED: Nov 16th
- test_create_watchlist_admin in tests.py was added
- pagination added to view: WatchlistAV
- WatchListSerializer was edited: overriding create, slug added, rev-count removed.
- adding one Python list per serializer model and replacing each automatic __all__/exclude declaration with its explicit list.
- WatchListDetailsAV renamed

UPDATED: Nov 16th
- WatchListSerializer updated: get_warning added
- WatchListAV updated: warning feature added
- len_name removed