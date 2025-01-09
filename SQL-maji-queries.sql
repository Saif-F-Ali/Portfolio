-- use md_water_services
-- show tables

-- select * from water_source
-- limit 100

-- select distinct type_of_water_source
-- from water_source

-- output:
-- type_of_water_source
-- well
-- tap_in_home_broken
-- tap_in_home
-- shared_tap
-- river

-- select * from visits
-- where time_in_queue > 500

-- select * 
-- from visits
-- join water_source
-- on water_source.source_id = visits.source_id
-- where time_in_queue > 500

-- output: type_of_water_source = shared tab

/*double join and conditions*/
-- select
-- 	water_source.type_of_water_source,
--     visits.visit_count,
--     subjective_quality_score
-- from water_quality
-- join visits
-- on visits.record_id = water_quality.record_id
-- join water_source
-- on water_source.source_id = visits.source_id
-- where type_of_water_source = 'tap_in_home'
-- and water_quality.subjective_quality_score = 10
-- and visits.visit_count > 1

-- output: there are no home taps with more than 1 visit count

-- select * from well_pollution
-- where biological > 0.01 and results = "Clean"

/*wild cards*/
-- select * from well_pollution
-- where description like 'Clean Bacteria%';

/*DML with case statement*/
-- update well_pollution
-- set description = case 
-- 	when description = 'Clean Bacteria: Giardia Lamblia' then 'Bacteria: Giardia Lamblia'
--     when description = 'Clean Bacteria: E. coli' then 'Bacteria: E. coli' end;

/*nested control*/
-- select * from well_pollution
-- where description like 'Clean Bacteria%' 
-- or (biological > 0.01 and results = "Clean");

-- update well_pollution
-- set results = 'Contaminated: Biological' 
-- where biological > 0.01 and results = "Clean"

-- select * from well_pollution
-- where biological > 0.01 and results = "Clean"

/*============================================================*/
/*adding emails*/
-- update employee
-- set email = concat(lcase(replace(employee_name, ' ', '.')), '@ndogowater.gov');

/*trimming spaces*/
-- select phone_number, length(employee.phone_number)
-- from employee;

-- update employee
-- set employee.phone_number = trim(phone_number);

-- select phone_number, length(employee.phone_number)
-- from employee;

/*employees per town*/
-- select
-- 	employee.town_name,
--     count(employee_name) as count
-- from employee
-- group by town_name;

/*visits per source*/
-- select 
-- 	location_id as loc,
-- 	source_id as src,
-- 	count(source_id) over(partition by source_id) as visit_cnt,
-- 	time_of_record as times
-- from visits
-- order by source_id, time_of_record asc;

/*visits per employee*/
-- select distinct
-- 	   visits.assigned_employee_id,
--     employee.employee_name,
--     count(visits.visit_count) over(partition by visits.assigned_employee_id) as visit0count
-- from visits
-- join employee
-- on employee.assigned_employee_id = visits.assigned_employee_id;

/* a view that gets visits start date, end date and count*/
-- drop view if exists visits_aggr;
-- create view visits_aggr as
-- select assigned_employee_id as id,
-- 	min(time_of_record) as start_date,
-- 	max(time_of_record) as end_date,
-- 	count(visit_count) as v_count
-- from visits
-- group by assigned_employee_id;

-- select * 
-- from visits_aggr;

/*combining tables 'employee' and 'visits_aggr' to create procedure full_surveyor_profile*/
-- drop procedure full_surveyor_profile;
-- delimiter \
-- create procedure full_surveyor_profile(emp_id int)
-- begin
-- select *
-- from employee
-- join visits_aggr
-- on employee.assigned_employee_id = visits_aggr.id
-- where employee.assigned_employee_id = emp_id;
-- end
-- \
-- call full_surveyor_profile(12);





