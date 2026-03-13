@testset "Truncated Tensor Algebra Tests :p2id" begin

    function axis_core_3tensor_QQ_p2id(_d)
        #C = zeros(QQ, _d, _d, _d)
        C = fill(zero(QQ), _d, _d, _d)
        for al in 1:_d, be in 1:_d, ga in 1:_d
            if al == be && be == ga
                C[al, be, ga] = QQ(1,6)
            elseif (al < be && be == ga) || (al == be && be < ga)
                C[al, be, ga] = QQ(1,2)
            elseif al < be && be < ga
                C[al, be, ga] = one(QQ)
            end
        end
        return C
    end

    function membrane_signature_QQ(m::Int, n::Int)
       Caxis_m = axis_core_3tensor_QQ_p2id(m)
       Caxis_n = axis_core_3tensor_QQ_p2id(n)

       d_m = size(Caxis_m, 1)
       d_n = size(Caxis_n, 1)
       d   = d_m * d_n

       #M = zeros(QQ, d, d, d)
       M = fill(zero(QQ), d, d, d)

       for i in 1:d_m, j in 1:d_m, k in 1:d_m
            for a in 1:d_n, b in 1:d_n, c in 1:d_n
                a2 = (i-1)*d_n + a
                b2 = (j-1)*d_n + b
                c2 = (k-1)*d_n + c
                M[a2, b2, c2] += Caxis_m[i,j,k] * Caxis_n[a,b,c]
            end
        end

        return M
    end

    @testset "Constructor TTA" begin
        d = 6
        k = 5
        T = TruncatedTensorAlgebra(QQ, d, k, sequence_type=:p2id)

        @test T == TruncatedTensorAlgebra(QQ, d, k, sequence_type=:p2id)
        @test sequence_type(T) == :p2id
        @test base_dimension(T) == d
        @test base_algebra(T) == QQ
        @test truncation_level(T) == k
    end

    @testset "Axis constructor QQ :p2id" begin
        d = 6
        T = TruncatedTensorAlgebra(QQ, d, 4, sequence_type=:p2id)

        m = 2
        n = 3
        shape = (m, n)

        Caxis_d = sig(T, :axis, shape=shape)

        @test parent(Caxis_d) == T
        @test zero(T) + zero(T) == zero(T)

        @test Caxis_d == sig(T, :axis, shape=shape, algorithm=:Chen)
        @test Caxis_d == sig(T, :axis, shape=shape, algorithm=:AFS19)

        for i in 2:m-1
            @test Caxis_d[i] == one(QQ)
            @test Caxis_d[i, i-1] == zero(QQ)
            @test Caxis_d[i, i] == QQ(1,2)
            @test Caxis_d[i, i+1] == one(QQ)
        end

        @test membrane_signature_QQ(m, n) == Caxis_d[:,:,:]

        @test zero(T) + Caxis_d == Caxis_d
        @test one(T) * Caxis_d == Caxis_d
        @test Caxis_d * one(T) == Caxis_d
        @test inv(Caxis_d) * Caxis_d == one(T)
        @test Caxis_d * inv(Caxis_d) == one(T)
        @test inv(inv(Caxis_d)) == Caxis_d
        @test Caxis_d^3 == Caxis_d * Caxis_d * Caxis_d
    end

    @testset "pwbln consistency matrix vs membrane" begin



        m = 2
        n = 3
        shape = (m, n)

        for coef in 1:20

            d = rand(2:8)
            T = TruncatedTensorAlgebra(QQ, d, 4, sequence_type=:p2id)
            
            coef=rand(-20:20,d,m*n)

            pwbln  = sig(T, :pwbln, coef=coef, shape=shape)
            pwbln2 = sig(T, :pwbln, coef=coef, shape=shape, algorithm=:LS26)

            @test pwbln == pwbln2

            membrane = Array{QQFieldElem}(undef, m, n, d)

            for di in 1:d
                cont = 0
                for i in 1:m, j in 1:n
                    cont += 1
                    membrane[i, j, di] = coef[di, cont]
                end
            end

            pwblnm  = sig(T, :pwbln, coef=membrane, shape=shape)
            pwblnm2 = sig(T, :pwbln, coef=membrane, shape=shape, algorithm=:LS26)

            @test pwblnm == pwblnm2
            @test pwbln  == pwblnm
            @test pwbln  == pwblnm2
        end
    end

    @testset "Axis constructor in TTA for polynomial rings :p2id" begin

       d = 6
       m = 2
       n = 3
       shape = (m, n)

       R, a = polynomial_ring(QQ, :a => (1:d, 1:d))
       T = TruncatedTensorAlgebra(R, d, 5, sequence_type=:p2id)

       Caxis_d = sig(T, :axis, shape=shape)

       @test parent(Caxis_d) == T
       @test zero(T) + zero(T) == zero(T)

       @test Caxis_d == sig(T, :axis, shape=shape, algorithm=:Chen)
       @test Caxis_d == sig(T, :axis, shape=shape, algorithm=:AFS19)

       for i in 2:m-1
           @test Caxis_d[i] == one(R)
           @test Caxis_d[i, i-1] == zero(R)
           @test Caxis_d[i, i] == R(1//2)
           @test Caxis_d[i, i+1] == one(R)
       end

       @test zero(T) + Caxis_d == Caxis_d
       @test one(T) * Caxis_d == Caxis_d
       @test Caxis_d * one(T) == Caxis_d
       @test inv(Caxis_d) * Caxis_d == one(T)
       @test Caxis_d * inv(Caxis_d) == one(T)
       @test inv(inv(Caxis_d)) == Caxis_d
  #    @test exp(log(Caxis_d)) == Caxis_d
       @test Caxis_d^3 == Caxis_d * Caxis_d * Caxis_d
 #     @test exp(log(Caxis_d) + log(Caxis_d)) == Caxis_d^2

    end


end
