"""Guards on how job-id ranges are turned into ids, and on who may vanish.

Two separate hazards, caught in two separate places:

1. `rm 1-100000000` -- one slipped keystroke -- used to build a hundred
   million integers (~7 GB, over a minute) before anything looked at the
   database. A range is now checked on its endpoint against the highest id
   the database holds, which costs a single MAX() query.
2. `rm 1-40` against a five-job database deleted five and reported
   "Removed 5 job(s)", which reads exactly like success. Every requested id
   must now exist or nothing is removed.
"""
import pytest

from parslbox.commands.helpers.job_id_parser import (
    format_job_ids,
    parse_job_ids,
    resolve_job_ids,
    validate_job_id_tokens,
)
from parslbox.database import database


@pytest.fixture
def db(tmp_path):
    """Jobs 1..5."""
    path = tmp_path / "jobs.db"
    database.initialize_database(path)
    for i in range(5):
        database.add_job(db_path=path, path=f"/w/j{i}", app="python", num_nodes=1,
                         ngpus=0, node_occupancy=1.0, tag="t", in_file=None,
                         ranks_per_node=1)
    return path


class TestMaxJobId:
    def test_reports_the_highest_id(self, db):
        assert database.get_max_job_id(db) == 5

    def test_an_empty_database_is_zero_not_none(self, tmp_path):
        path = tmp_path / "empty.db"
        database.initialize_database(path)
        assert database.get_max_job_id(path) == 0

    def test_a_database_that_does_not_exist_is_zero(self, tmp_path):
        assert database.get_max_job_id(tmp_path / "nope.db") == 0

    def test_it_tracks_the_highest_id_not_the_row_count(self, db):
        database.remove_jobs_by_id(db, [4, 5])
        assert database.get_max_job_id(db) == 3


class TestRangesAreBoundedBeforeExpansion:
    def test_a_range_past_the_last_job_is_refused(self, db):
        with pytest.raises(ValueError, match="highest job id in the database"):
            resolve_job_ids(["1-100000000"], db)

    def test_refusal_is_immediate_not_after_building_the_range(self, db):
        """If the endpoint check ever moves after the expansion this takes
        the better part of a minute and several GB instead of microseconds."""
        import time
        start = time.perf_counter()
        with pytest.raises(ValueError):
            resolve_job_ids(["1-100000000"], db)
        assert time.perf_counter() - start < 0.5

    def test_the_message_names_the_token_and_the_ceiling(self, db):
        with pytest.raises(ValueError) as exc:
            resolve_job_ids(["3-900"], db)
        assert "3-900" in str(exc.value) and "5" in str(exc.value)

    def test_an_empty_database_says_so_rather_than_naming_zero(self, tmp_path):
        path = tmp_path / "empty.db"
        database.initialize_database(path)
        with pytest.raises(ValueError, match="no jobs"):
            resolve_job_ids(["1-10"], path)

    def test_a_range_ending_exactly_at_the_last_job_is_fine(self, db):
        assert resolve_job_ids(["1-5"], db) == [1, 2, 3, 4, 5]

    def test_a_bare_id_past_the_last_job_is_left_alone(self, db):
        """Only ranges can blow up. Commands report unknown ids themselves,
        and `info 1 99` warning about 99 is more useful than refusing both."""
        assert resolve_job_ids(["1", "99999"], db) == [1, 99999]

    def test_ids_may_be_ints_as_well_as_strings(self, db):
        assert resolve_job_ids([3, "1-2"], db) == [1, 2, 3]

    def test_the_ceiling_is_optional(self):
        """`parse_job_ids` is still usable where no database is in reach."""
        assert parse_job_ids(["1-5"]) == [1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            parse_job_ids(["1-5"], max_range_end=3)


class TestTokenSyntaxCheckedWithoutExpanding:
    """What the MCP schema validator uses, having no database to bound with."""

    def test_a_huge_range_passes_syntax_and_stays_a_token(self):
        assert validate_job_id_tokens(["1-100000000"]) is None

    def test_a_backwards_range_is_still_caught(self):
        with pytest.raises(ValueError, match="start must be <= end"):
            validate_job_id_tokens(["5-1"])

    def test_a_non_numeric_token_is_still_caught(self):
        with pytest.raises(ValueError):
            validate_job_id_tokens(["abc"])


class TestRemoveRejectsIdsThatDoNotExist:
    def test_a_gap_stops_the_whole_delete(self, db):
        database.remove_jobs_by_id(db, [3])
        with pytest.raises(ValueError, match="No job with id 3"):
            database.remove_jobs_by_id(db, [1, 2, 3, 4])
        assert len(database.get_jobs(db)) == 4

    def test_missing_ids_are_compressed_not_listed_one_by_one(self, db):
        database.remove_jobs_by_id(db, [2, 3, 4])
        with pytest.raises(ValueError) as exc:
            database.remove_jobs_by_id(db, [1, 2, 3, 4, 5])
        assert "2-4" in str(exc.value)
        assert "2, 3, 4" not in str(exc.value)

    def test_the_compression_is_the_shared_formatter(self, db):
        """Same renderer the rest of pbx uses, so a wide gap cannot turn into
        a wall of integers in one layer and a range in another."""
        for jid in (2, 3, 4):
            database.remove_jobs_by_id(db, [jid])
        with pytest.raises(ValueError) as exc:
            database.remove_jobs_by_id(db, [1, 2, 3, 4, 5])
        assert format_job_ids([2, 3, 4]) in str(exc.value)

    def test_every_id_present_removes_them_all(self, db):
        assert database.remove_jobs_by_id(db, [1, 2, 3]) == 3

    def test_an_empty_request_removes_nothing_and_does_not_raise(self, db):
        assert database.remove_jobs_by_id(db, []) == 0
        assert len(database.get_jobs(db)) == 5


class TestIdsTooLargeForSqlite:
    """A bare id is not bounded by the database's height, so an extra digit
    used to reach the sqlite3 driver and raise OverflowError -- which no CLI
    command catches. It now fails like any other bad id."""

    def test_an_id_past_sqlites_integer_range_is_refused(self, db):
        from parslbox.commands.helpers.job_id_parser import MAX_JOB_ID
        with pytest.raises(ValueError, match="out of range"):
            resolve_job_ids([str(MAX_JOB_ID + 1)], db)

    def test_the_largest_storable_id_is_still_accepted(self, db):
        from parslbox.commands.helpers.job_id_parser import MAX_JOB_ID
        assert resolve_job_ids([str(MAX_JOB_ID)], db) == [MAX_JOB_ID]

    def test_a_range_ending_past_it_is_refused_too(self, db):
        with pytest.raises(ValueError, match="out of range"):
            resolve_job_ids(["1-" + "9" * 30], db)

    def test_the_syntax_only_check_refuses_it_as_well(self):
        """The MCP validator must not pass one through to the driver either."""
        with pytest.raises(ValueError, match="out of range"):
            validate_job_id_tokens(["9" * 30])


def _seed(path, n):
    """n jobs, inserted directly -- add_job per row is too slow at this size."""
    import sqlite3
    database.initialize_database(path)
    con = sqlite3.connect(path)
    con.executemany("INSERT INTO jobs (job_id, path, app, status) VALUES (?,?,?,?)",
                    [(i, f"/w/{i}", "python", "Ready") for i in range(1, n + 1)])
    con.commit()
    con.close()
    return path


class TestWideRequestsExceedSqlitesParameterLimit:
    """SQLite binds at most ~32k host parameters per statement, so an
    `IN (?,?,...)` built from a job-id list failed with "too many SQL
    variables" on a database that had simply grown -- a legitimate
    `rm 1-33000`, not a typo. The ceiling guard approves such a range, so
    this is the next thing the request hits.
    """

    N = 33000

    def test_removing_every_job_in_a_large_database(self, tmp_path):
        path = _seed(tmp_path / "big.db", self.N)
        assert database.remove_jobs_by_id(path, list(range(1, self.N + 1))) == self.N

    def test_reading_every_job_in_a_large_database(self, tmp_path):
        path = _seed(tmp_path / "big.db", self.N)
        assert len(database.get_jobs_by_ids(path, list(range(1, self.N + 1)))) == self.N

    def test_updating_every_job_in_a_large_database(self, tmp_path):
        path = _seed(tmp_path / "big.db", self.N)
        assert database.update_jobs(path, list(range(1, self.N + 1)),
                                    status="Done") == self.N

    def test_chunking_does_not_disturb_the_documented_id_order(self, tmp_path):
        """get_jobs_by_ids promises ascending ids; it used to get that from
        a single ORDER BY, which no longer spans the whole request."""
        path = _seed(tmp_path / "big.db", 5)
        got = [j["job_id"] for j in database.get_jobs_by_ids(path, [5, 1, 3])]
        assert got == [1, 3, 5]


class TestTheExistenceCheckAndTheDeleteAreAtomic:
    """Several orchestrators share one database. A writer slipping between
    the check and the DELETE reinstates the partial delete this function
    exists to refuse."""

    def test_a_concurrent_writer_cannot_open_the_window(self, db):
        import sqlite3
        import threading
        import time

        note = []
        started = threading.Event()

        def intruder():
            started.wait()
            con = sqlite3.connect(db, timeout=0.4)
            try:
                con.execute("DELETE FROM jobs WHERE job_id = 5")
                con.commit()
                note.append("wrote")
            except sqlite3.OperationalError:
                note.append("locked out")
            finally:
                con.close()

        real = database._id_chunks

        def slow_chunks(job_ids, con, room=0):
            """Hold the window open between the check and the DELETE."""
            started.set()
            time.sleep(0.6)
            yield from real(job_ids, con, room)

        thread = threading.Thread(target=intruder)
        thread.start()
        database._id_chunks = slow_chunks
        try:
            removed = database.remove_jobs_by_id(db, [3, 4, 5])
        finally:
            database._id_chunks = real
            thread.join(timeout=5)

        assert note == ["locked out"], f"intruder was not blocked: {note}"
        assert removed == 3
        assert [j["job_id"] for j in database.get_jobs(db)] == [1, 2]


class TestTheCountIsRowsNotRequests:
    def test_duplicate_ids_are_counted_once(self, db):
        """`api.remove_jobs` does not dedupe, so the count must come from the
        rows actually deleted rather than the length of the request."""
        assert database.remove_jobs_by_id(db, [3, 3, 4]) == 2


class TestChunkSizeComesFromTheDriver:
    """The per-statement parameter cap is a compile-time SQLite setting, so
    hardcoding it means guessing low forever or breaking on an old build."""

    def _con(self, limit):
        import sqlite3
        con = sqlite3.connect(":memory:")
        con.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
        return con

    def test_no_chunk_exceeds_what_the_connection_allows(self):
        con = self._con(10)
        chunks = list(database._id_chunks(list(range(100)), con))
        assert max(len(c) for c in chunks) == 10
        assert sum(len(c) for c in chunks) == 100

    def test_room_for_the_rest_of_the_statement_is_subtracted(self):
        """`update_jobs` spends parameters on its SET clause before the ids."""
        con = self._con(10)
        chunks = list(database._id_chunks(list(range(100)), con, room=4))
        assert max(len(c) for c in chunks) == 6

    def test_a_budget_smaller_than_the_room_still_makes_progress(self):
        con = self._con(3)
        chunks = list(database._id_chunks([1, 2, 3], con, room=99))
        assert [len(c) for c in chunks] == [1, 1, 1]

    def test_a_generous_limit_is_actually_used(self):
        con = self._con(32766)
        assert len(list(database._id_chunks(list(range(5000)), con))) == 1
