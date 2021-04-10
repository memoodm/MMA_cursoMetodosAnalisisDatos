
file = "
	..\\MMA_cursoMetodosAnalisisDatos
	\\Guillermo De Mendoza
	\\Mineria de datos - semestre 4
	\\Tarea4
	\\ETL-out-variablesNumeric.txt"
datos<-read.table(file, header=T, row.names=1)
rec = glm(isInTop100~.,family="binomial", data=datos)
rec


