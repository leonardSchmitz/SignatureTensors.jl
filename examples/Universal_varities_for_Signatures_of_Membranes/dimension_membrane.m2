needsPackage "NumericalImplicitization";

d=2
m=2
n=2
f = (i,j) -> if i<=j then if i==j then 1/2 else 1 else 0;

orb_dim = (m,n,d) -> (
    R := CC[s_(1,1)..s_(m*n,d)];
    S := genericMatrix(R, s_(1,1), m*n, d);

    Am =   matrix table(m, m, (i,j) ->  (j+1) / (i + j+2));
    An =  matrix table(n, n, (i,j) ->  (j+1) / (i + j+2));
   

    A :=  tensor(Am, An);               -- matriz C
    A2 :=  tensor(Am,transpose An);    -- matriz C2


    Q1 := transpose S * A * S;
    Q2 := transpose S * A2 * S;


    getAllEntries = M -> (
            flatten for i from 0 to d-1 list (
                   for j from 0 to d-1 list (
                         M_(i,j)
                         )
                        )
                );
    outList = join(getAllEntries(Q1), getAllEntries(Q2));



    phi := map(R, CC[x_0..x_(#outList - 1)], outList);


    numericalImageDim(phi, ideal 0_R)
);

-- m n d
orb_dim(2,4,5)
