function T = nmytypes(dt) 
  switch dt 
    case 'double' 
      T.p_XY = double([]);
      T.p_X_p_Y = double([]);
      T.MI = double([]);
      T.score_per_feature = double([]);
     case 'single' 
      T.p_XY = single([]);
      T.p_X_p_Y = single([]);
      T.MI = single([]);
      T.score_per_feature = single([]);
     case 'fixed4_1' 
      T.p_XY = fi([],true, 4, 1);
      T.p_X_p_Y = fi([],true, 4, 1);
      T.MI = fi([],true, 4, 1);
      T.score_per_feature = fi([],true, 4, 1);
     case 'fixed4_2' 
      T.p_XY = fi([],true, 4, 2);
      T.p_X_p_Y = fi([],true, 4, 2);
      T.MI = fi([],true, 4, 2);
      T.score_per_feature = fi([],true, 4, 2);
    case 'fixed4_3' 
      T.p_XY = fi([],true, 4, 3);
      T.p_X_p_Y = fi([],true, 4, 3);
      T.MI = fi([],true, 4, 3);
      T.score_per_feature = fi([],true, 4, 3);
    case 'fixed8_2' 
      T.p_XY = fi([],true, 8, 2);
      T.p_X_p_Y = fi([],true, 8, 2);
      T.MI = fi([],true, 8, 2);
      T.score_per_feature = fi([],true, 8, 2);
    case 'fixed8_4' 
      T.p_XY = fi([],true, 8, 4);
      T.p_X_p_Y = fi([],true, 8, 4);
      T.MI = fi([],true, 8, 4);
      T.score_per_feature = fi([],true, 8, 4);
    case 'fixed8_6' 
      T.p_XY = fi([],true, 8, 6);
      T.p_X_p_Y = fi([],true, 8, 6);
      T.MI = fi([],true, 8, 6);
      T.score_per_feature = fi([],true, 8, 6);
    case 'fixed16_4'
      T.p_XY = fi([],true, 16, 4);
      T.p_X_p_Y = fi([],true, 16, 4);
      T.MI = fi([],true, 16, 4);
      T.score_per_feature = fi([],true, 16, 4);
    case 'fixed16_8'
      T.p_XY = fi([],true, 16, 8);
      T.p_X_p_Y = fi([],true, 16, 8);
      T.MI = fi([],true, 16, 8);
      T.score_per_feature = fi([],true, 16, 8);
    case 'fixed16_12'
      T.p_XY = fi([],true, 16, 12);
      T.p_X_p_Y = fi([],true, 16, 12);
      T.MI = fi([],true, 16, 12);
      T.score_per_feature = fi([],true, 16, 12);
    case 'fixed32_8' 
      T.p_XY = fi([],true, 32, 8);
      T.p_X_p_Y = fi([],true, 32, 8);
      T.MI = fi([],true, 32, 8);
      T.score_per_feature = fi([],true, 32, 8);
    case 'fixed32_16' 
      T.p_XY = fi([],true, 32, 16);
      T.p_X_p_Y = fi([],true, 32, 16);
      T.MI = fi([],true, 32, 16);
      T.score_per_feature = fi([],true, 32, 16);
    case 'fixed32_24' 
      T.p_XY = fi([],true, 32, 24);
      T.p_X_p_Y = fi([],true, 32, 24);
      T.MI = fi([],true, 32, 24);
      T.score_per_feature = fi([],true, 32, 24);
  end 
end 