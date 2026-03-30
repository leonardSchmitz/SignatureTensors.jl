
@testset "Truncated Tensor Algebra Tests :iis" begin

    @testset "Constructor TTA" begin
       d = 6;        # path dimension
       k = 5;        # truncation level
       T = TruncatedTensorAlgebra(QQ,d,k);
       @test T == TruncatedTensorAlgebra(QQ,d,k,sequence_type=:iis)
       @test sequence_type(T) == :iis
       @test base_dimension(T) == d
       @test base_algebra(T) == QQ
       @test truncation_level(T) == k
    end

    function axis_core_3tensor_QQ(_d)
        C=fill(zero(QQ), _d, _d, _d)
        #C = zeros(QQ,_d,_d,_d);
        for al in (1:_d)
          for be in (1:_d)
            for ga in (1:_d)
              if al == be && be == ga
                C[al,be,ga] = QQ(1,6)
              end
              if (al < be && be == ga)||(al == be && be < ga)
                C[al,be,ga] = QQ(1,2)
              end
              if al < be && be < ga
                C[al,be,ga] = one(QQ)
              end
            end
          end
        end
        return C
    end

    @testset "Axis constructor in TTA for QQ" begin
        d = 6;
        T = TruncatedTensorAlgebra(QQ,d,4);
        Caxis_d = sig(T,:axis);
        @test parent(Caxis_d) == T
        @test zero(T) + zero(T) == zero(T)
        @test Caxis_d == sig(T,:axis,algorithm=:Chen)
        @test Caxis_d == sig(T,:axis,algorithm=:AFS19) 
        for i in (2:d-1)
            @test Caxis_d[i] == one(QQ)
            @test Caxis_d[i,i-1] == zero(QQ)
            @test Caxis_d[i,i] == QQ(1,2) 
            @test Caxis_d[i,i+1] == one(QQ)
        end 
        @test axis_core_3tensor_QQ(d) == Caxis_d[:,:,:]
        @test zero(T) + Caxis_d == Caxis_d
        @test one(T)*Caxis_d == Caxis_d
        @test Caxis_d*one(T) == Caxis_d
        @test inv(Caxis_d)*Caxis_d == one(T)
        @test Caxis_d*inv(Caxis_d) == one(T)
        @test inv(inv(Caxis_d)) == Caxis_d
        @test exp(log(Caxis_d)) == Caxis_d
        @test Caxis_d^3 == Caxis_d*Caxis_d*Caxis_d
        @test exp(log(Caxis_d) + log(Caxis_d)) == Caxis_d^2
    end

    @testset "Axis constructor in TTA for polynomial rings" begin
        d = 6;
        R, a = polynomial_ring(QQ, :a => (1:d, 1:d));
        T = TruncatedTensorAlgebra(R,d,5);
        Caxis_d = sig(T,:axis);
        @test parent(Caxis_d) == T
        @test zero(T) + zero(T) == zero(T)
        @test Caxis_d == sig(T,:axis,algorithm=:Chen)
        @test Caxis_d == sig(T,:axis,algorithm=:AFS19) 
        for i in (2:d-1)
            @test Caxis_d[i] == one(R)
            @test Caxis_d[i,i-1] == zero(R)
            @test Caxis_d[i,i+1] == one(R)
        end 
        @test zero(T) + Caxis_d == Caxis_d
        @test one(T)*Caxis_d == Caxis_d
        @test Caxis_d*one(T) == Caxis_d
        @test inv(Caxis_d)*Caxis_d == one(T)
        @test Caxis_d*inv(Caxis_d) == one(T)
        @test inv(inv(Caxis_d)) == Caxis_d
        @test exp(log(Caxis_d)) == Caxis_d
        @test Caxis_d^3 == Caxis_d*Caxis_d*Caxis_d
        @test exp(log(Caxis_d) + log(Caxis_d)) == Caxis_d^2
    end

    @testset "Piecewise monomial in TTA for QQ" begin 
        d = 6;
        for r in [0,1]
          for m in [[1,1,2],[2,3],[2,2,2],[2,1,3,1,1]]
              l = length(m);
              M = sum(m);
              n = sum(m) - r*(l-1); 
              T = TruncatedTensorAlgebra(QQ,n,4); 
              S = sig(T,:pwmon,composition=m,regularity=r);
              @test S == sig(T,:pwmon,composition=m,regularity=r,algorithm=:Chen); 
              @test S == sig(T,:pwmon,composition=m,regularity=r,algorithm=:ALS26); 
              @test zero(T) + S == S
              @test one(T)*S == S
              @test S*one(T) == S
              @test inv(S)*S == one(T)
              @test S*inv(S) == one(T)
              @test inv(inv(S)) == S
              @test exp(log(S)) == S
              @test S^3 == S*S*S
              @test exp(log(S) + log(S)) == S^2
          end
        end
        r = 2;
        for m in [[2,3,2],[3,3],[4,3]]
              l = length(m);
              M = sum(m);
              n = sum(m) - r*(l-1); 
              T = TruncatedTensorAlgebra(QQ,n,4); 
              S = sig(T,:pwmon,composition=m,regularity=r);
              @test S == sig(T,:pwmon,composition=m,regularity=r,algorithm=:Chen); 
              @test S == sig(T,:pwmon,composition=m,regularity=r,algorithm=:ALS26); 
              @test zero(T) + S == S
              @test one(T)*S == S
              @test S*one(T) == S
              @test inv(S)*S == one(T)
              @test S*inv(S) == one(T)
              @test inv(inv(S)) == S
              @test exp(log(S)) == S
              @test S^3 == S*S*S
              @test exp(log(S) + log(S)) == S^2
        end
    end

    @testset "Pwln constructor in TTA" begin
        d = 6; k = 4; number_tests = 5;
        T = TruncatedTensorAlgebra(QQ,d,k);
        ms = rand((1:7),number_tests); # maximal 7 number of segments
        As = [QQ.(rand((-20:20),d,m)) for m in ms];
        for A in As
          m = size(A)[2]  # number of segments in A
          S = sig(T,:pwln,coef=A); 
          @test S == sig(T,:pwln,coef=A,algorithm=:Chen); 
          @test S == sig(T,:pwln,coef=A,algorithm=:congruence); 
          @test S == sig(T,:pwln,coef=A,algorithm=:LS26);
          C = sig(TruncatedTensorAlgebra(QQ,m,k),:axis); # axis tensor in dim m
          @test S == A*C
          @test zero(T) + S == S
          @test one(T)*S == S
          @test S*one(T) == S
          @test inv(S)*S == one(T)
          @test S*inv(S) == one(T)
          @test inv(inv(S)) == S
          @test exp(log(S)) == S
          @test S^3 == S*S*S
          @test exp(log(S) + log(S)) == S^2
        end 
    end    
    
    @testset "Converting base algebra via matrix tensor congruence" begin
        d = 6; m = 5; k = 4;
        R, a = polynomial_ring(QQ, :a => (1:d, 1:m));
        TmQQ = TruncatedTensorAlgebra(QQ,m,k);
        TdR = TruncatedTensorAlgebra(R,d,k);
        TmR = TruncatedTensorAlgebra(R,m,k);
        C = sig(TmQQ,:axis);
        S = sig(TdR,:pwln,coef=a);
        @test S == a*C
        A = QQ.(rand((-20:20),d,m));
        @test A*sig(TmR,:axis) ==  sig(TdR,:pwln,coef=A);
    end 
end