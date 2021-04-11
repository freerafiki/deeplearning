// GLOBAL VARIABLES 
var weekprefix = "week";
var symbolprefix = "symb";
var openCl = "open";
var closedCl = "closed";
var rotatedCl = "rotated";

// SHOW THE CLICKED WEEK AND HIDE THE OTHERS
function toggleWeek(num) {
	var week = weekprefix.concat(num);
	var symbol = symbolprefix.concat(num);
	var weekEl = document.getElementById(week);
	if (weekEl.classList.contains(openCl)) {
		closeWeek(weekEl);
		document.getElementById(symbol).classList.remove(rotatedCl);
	}
	else if (weekEl.classList.contains(closedCl)) {
		openWeek(weekEl);
		document.getElementById(symbol).classList.add(rotatedCl);
	}
	else {
		alert("eh?");
	}
	
}

function closeWeek(weekEl) {
	weekEl.style.display = "none";
	weekEl.classList.remove(openCl);
	weekEl.classList.add(closedCl);
}

function openWeek(weekEl) {
	weekEl.style.display = "inline";
	weekEl.classList.remove(closedCl);
	weekEl.classList.add(openCl);
}

function closeAllWeeks() {
	var weeks = document.getElementsByClassName("weekcontent");
	for (i=0; i<weeks.length; i++) {
		closeWeek(weeks[i]);	
	}
}