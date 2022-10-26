create table constants.DAY_BAR_DICTIONARY_TABLE
(
	ticker varchar(10) primary key,
	table_name varchar(21),
	years integer[],
	last_date timestamp with time zone,
	bars_count integer,
	memory real
);

create table constants.DAY_BAR_RUSSEL_PRESENCE_TABLE
(yr smallint,
present varchar(10)[],
absent varchar(10)[])
;


truncate bars_1_day_1dictionary;