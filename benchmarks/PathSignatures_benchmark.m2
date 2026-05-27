loadPackage("PathSignatures",Reload=>true)
debug PathSignatures

getParts = a -> (
	ts := terms a;
	applyValues(partition(i->degree i,ts),sum)
)

truncMul = method()
truncMul(NCRingElement, NCRingElement, ZZ) := (a,b,k) ->
(
	ap := getParts(a);
	bp := getParts(b);
	sum flatten toList apply(0..k, i-> toList apply(0..i, j-> (
		f1 := 0;
		f2 := 0;
		if(ap#?j) then (f1 = ap#j);
		if(bp#?(i-j)) then (f2 = bp#(i-j));
		f1 * f2)
		))
)

sigT = method()
sigT(Path,ZZ,NCRing) := (X, h, R) -> (
    nop := X.numberOfPieces;
    d := X.dimension;
    if(h == 0) then return 1_R;
    if(nop == 0) then return 0;
    ws = flatten toList apply(1..h, i-> toList((i:1)..(i:d)));
    sigs = apply(nop, i -> (
	vec = apply(X.pieces#i,i-> i#0#1);
	seg = sum(length(vec),i->vec#i * [i+1]_R);
        -sum(0..h, i -> tensorExp(seg, i))
	--1_R + sum(ws, w -> 1/((length w)!) * product(toList w, j->vec#(j-1)) * (new Array from w)_R)
    ));
    fold(sigs, (i,j) -> truncMul(i,j,h))
)

d=10;
k=3;
Ad = wordAlgebra(d);
M = matrix toList apply(1..d,i->toList apply(1..d, i->random(-20,20)))
X = pwLinPath(M);
time mT = sigT(X,k,Ad);
time mT = sig(X,k,Ad);

