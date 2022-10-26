create table tz_test (
	dt timestamp with time zone,
	d1 timestamp with time zone,
	d2 timestamp with time zone,
	d3 timestamp with time zone,
	d10 timestamp with time zone,
	dJan4 timestamp with time zone,
	dJuly1 timestamp with time zone
-- 	r1 timestamp with time zone,
-- 	r2 timestamp with time zone,
-- 	r3 timestamp with time zone,
-- 	r10 timestamp with time zone,
-- 	rJan4 timestamp with time zone,
-- 	rJuly1 timestamp with time zone
);
commit;
select * from tz_test;