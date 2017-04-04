class TimGraph: public InfGraph
{
    public:
        TimGraph(string folder, string graph_file):InfGraph(folder, graph_file ){
        }
      
        double KptEstimation()
        {
            Timer t(1, "step1");
            
            double lb=1/2.0;
            double c=0;
            while(true){
                int loop= (6 * log(n)  +  6 * log(log(n)/ log(2)) )* 1/lb  ;
                c=0; //sum
                IF_TRACE(int64 now=rdtsc());
                double sumMgTu=0;
                for(int i=0; i<loop; i++){
                    int u=rand()%n; //  
                    ASSERT(u>=0);
                    ASSERT(u<n);
                    double MgTu=(double)BuildHypergraphNode(u, 0, false); //width of set
                    double pu=MgTu/m;
                    sumMgTu+=MgTu;
                    c+=1-pow((1-pu), k);
                }
                c/=loop;
                if(c>lb) break;
                lb /= 2;
            }
            return c * n/2; 
        }

        void RefineKPT(double epsilon, double ept){
            Timer t(2, "step2" );
            ASSERT(ept > 0);
            int64 R = (2 + epsilon) * ( n * log(n) ) / ( epsilon * epsilon * ept);
            BuildHypergraphR(R);
        }
        
        double EstimateOPT(double epsilon){
            // KPT estimation
            double kpt_star;
            kpt_star=KptEstimation();
            // printf("kpt estimation: %.2f\n",kpt_star);

            // Refine KPT
            double eps_prime;
            eps_prime=5*pow(sqr(epsilon)/(k+1), 1.0/3.0);
            RefineKPT(eps_prime, kpt_star);
            BuildSeedSet();
            double kpt=InfluenceHyperGraph();
            kpt/=1+eps_prime;
            double kpt_plus = max(kpt, kpt_star);
            // printf("kpt refined: %.2f (%.2f, %.2f)\n", kpt_plus, kpt, kpt_star);
        
            // cout<<"TIME kpt Estimation: " << Timer::timeUsed[1]/TIMES_PER_SEC << "s" <<endl;
            // cout<<"TIME kpt refine: " << Timer::timeUsed[2]/TIMES_PER_SEC << "s" <<endl;
            return kpt_plus;
        }
};

